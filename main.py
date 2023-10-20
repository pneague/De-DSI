import io
import sys
import os
from asyncio import run, sleep
from dataclasses import dataclass
import threading
import queue
import torch
import time

from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from ipv8.types import Peer
from ipv8.util import run_forever
from ipv8_service import IPv8

from simple_term_menu import TerminalMenu
from ltr import LTR
from utils import *

# Enhance normal dataclasses for IPv8 (see the serialization documentation)
dataclass = overwrite_dataclass(dataclass)

@dataclass(msg_id=1)  # The value 1 identifies this message and must be unique per community
class UpdateModel:
    id: bytes
    fragment: int
    total: int
    data: bytes


class LTRCommunity(Community):
    community_id = b'\x9d\x10\xaa\x8c\xfa\x0b\x19\xee\x96\x8d\xf4\x91\xea\xdc\xcb\x94\xa7\x1d\x8b\x9c'

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.add_message_handler(UpdateModel, self.on_message)
        self.input_queue = queue.Queue()
        self.ready_for_input = threading.Event()
        self.packets: list[UpdateModel] = []

    def input_thread(self):
        while True:
            self.ready_for_input.wait()
            query = input(f"{colorize('QUERY', 'green')}:")
            self.input_queue.put(query)
            self.ready_for_input.clear()

    def started(self) -> None:
        print('Indexing (please wait)...')
        self.ltr = LTR()

        async def app() -> None:
            threading.Thread(target=self.input_thread, daemon=True).start()
            while True:
                self.ready_for_input.set()
                while self.input_queue.empty():
                    await sleep(0.1)

                query = self.input_queue.get()
                results = self.ltr.query(query)
                terminal_menu = TerminalMenu(results)
                selected_res = terminal_menu.show()
                print(f"{colorize('RESULT', 'blue')}:", results[selected_res])
                await sleep(0)
                self.ltr.on_result_selected(query, selected_res)
                for peer in self.get_peers():
                    model_bf = self.ltr.serialize_model()
                    chunks = split(model_bf, 8192)
                    _id = os.urandom(16)
                    preprint(f'sending update (packet 0/{len(chunks)})')
                    for i, chunk in enumerate(chunks):
                        reprint(f'\rsending update (packet {i+1}/{len(chunks)})')
                        self.ez_send(peer, UpdateModel(_id, i+1, len(chunks), chunk))
                        time.sleep(0.01)
                    print('\n')

        self.register_task("app", app, delay=0)


    @lazy_wrapper(UpdateModel)
    def on_message(self, peer: Peer, payload: UpdateModel) -> None:
        self.packets.append(payload)
        packets = [x for x in self.packets if x.id == payload.id]
        if len(packets) == payload.total:
            model_bf = io.BytesIO()
            for packet in sorted(packets, key=lambda x: x.fragment):
                model_bf.write(packet.data)
            self.packets = list(filter(lambda x: x.id != payload.id, self.packets))
            model_bf.seek(0)
            self.ltr.apply_updates(torch.load(model_bf))
            print('model updated')

async def start_communities() -> None:
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("my peer", "medium", f"certs/ec{sys.argv[1]}.pem")
    builder.add_overlay("LTRCommunity", "my peer",
                        [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                        default_bootstrap_defs, {}, [('started',)])
    await IPv8(builder.finalize(),
                extra_communities={'LTRCommunity': LTRCommunity}).start()
    await run_forever()

run(start_communities())