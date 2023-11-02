import io
import sys
import os
from asyncio import run, sleep
from dataclasses import dataclass
import threading
import queue
import time
import argparse
import random
import torch
from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from ipv8.types import Peer
from ipv8.util import run_forever
from ipv8_service import IPv8

from simple_term_menu import TerminalMenu
from p2p_ol2r.ltr import LTR
from p2p_ol2r.utils import *

# Enhance normal dataclasses for IPv8 (see the serialization documentation)
dataclass = overwrite_dataclass(dataclass)

@dataclass(msg_id=1)  # The value 1 identifies this message and must be unique per community
class UpdateModel:
    id: bytes
    fragment: int
    total: int
    data: bytes


class LTRCommunity(Community):
    community_id = b'\x9d\x10\xaa\x8c\xfa\x0b\x19\xee\x96\x8d\xf4\x91\xea\xdc\xcb\x94\xa7\x1d\x8c\x00'

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        if args.quantize:
            self.community_id = self.community_id[:-1] + bytes([0x01])
        self.add_message_handler(UpdateModel, self.on_message)
        self.input_queue = queue.Queue()
        self.ready_for_input = threading.Event()
        self.packets: list[UpdateModel] = []

    def input_thread(self):
        while True:
            self.ready_for_input.wait()
            query = input(f"\r{fmt('QUERY', 'purple')}: ")
            self.input_queue.put(query)
            self.ready_for_input.clear()

    def started(self) -> None:
        print('Indexing (please wait)...')
        self.ltr = LTR(args.k, args.quantize)

        if args.simulation:
            print(fmt('Enter query for simulation', 'yellow'))
            query = input(f"\r{fmt('QUERY', 'purple')}: ")

            print(fmt('Consecutively select results for simulation from best to worst', 'yellow'))

            ranked_result_ids = [] # selected results ordered by rank (best to worst)
            results = self.ltr.query(query)
            remaining_results = results.copy()

            for rank in range(len(results)):
                selected_res = None
                while selected_res is None:
                    terminal_menu = TerminalMenu(remaining_results.values())
                    selected_res = terminal_menu.show()

                selected_id, selected_title = list(remaining_results.items())[selected_res]
                print(fmt(f'#{rank+1}', 'blue') + f': {selected_title}')
                ranked_result_ids.append(selected_id)
                remaining_results.pop(selected_id)

            # For result #1, e.g., simulate sim_epochs=100 clicks, for result #2, simulate 90 clicks, etc.
            selected_results = []
            sim_epochs = int(input(f"\r{fmt('Number of epochs on #1 (e.g., 1000)', 'yellow')}: "))
            sim_epoch_diff = int(input(f"\r{fmt('Deduction per rank (e.g., 100)', 'yellow')}: "))
            for i in range(len(ranked_result_ids)):
                if sim_epoch_diff <= 0: break
                selected_results += [list(results.keys()).index(ranked_result_ids[i])] * (sim_epochs - i*sim_epoch_diff)
            random.shuffle(selected_results)
            
            print(fmt(f'Training model on simulation ({len(selected_results)} epochs)...', 'gray'))

            with silence():
                for res in selected_results:
                    self.ltr.on_result_selected(query, ranked_result_ids, res)
            
            inferred_ranking = list(self.ltr.query(query).keys())
            print(ranked_result_ids, inferred_ranking)
            print(fmt(f'nDCG: {round(ndcg(ranked_result_ids, inferred_ranking), 3)}', 'yellow'))
            print(fmt(f'Random nDCG: {round(ndcg(random.sample(ranked_result_ids, len(ranked_result_ids)), inferred_ranking), 3)}', 'yellow'))

        async def app() -> None:
            threading.Thread(target=self.input_thread, daemon=True).start()

            while True:
                self.ready_for_input.set()
                while self.input_queue.empty():
                    await sleep(0.1)

                query = self.input_queue.get()
                results = self.ltr.query(query)

                terminal_menu = TerminalMenu(results.values())
                selected_res = terminal_menu.show()
                if selected_res is None: continue
                print(f"{fmt('RESULT', 'blue')}:", list(results.values())[selected_res])
                await sleep(0)

                self.ltr.on_result_selected(query, list(results.keys()), selected_res)
                model_bf = self.ltr.serialize_model()
                chunks = split(model_bf, 8192)
                for peer in self.get_peers():
                    _id = os.urandom(16)
                    print(fmt(f'Sending update (packet 0/{len(chunks)})', 'gray'), end='')
                    for i, chunk in enumerate(chunks):
                        print(fmt(f'\rSending update (packet {i+1}/{len(chunks)})', 'gray'), end='')
                        self.ez_send(peer, UpdateModel(_id, i+1, len(chunks), chunk))
                        time.sleep(0.01)
                    print()

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
            model = torch.load(model_bf)
            self.ltr.apply_updates(model)

async def start_communities() -> None:
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("my peer", "medium", f"certs/ec{args.id}.pem")
    builder.add_overlay("LTRCommunity", "my peer",
                        [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                        default_bootstrap_defs, {}, [('started',)])
    await IPv8(builder.finalize(),
                extra_communities={'LTRCommunity': LTRCommunity}).start()
    await run_forever()

parser = argparse.ArgumentParser(prog='Peer-to-Peer Online Learning-to-Rank')
parser.add_argument('id', help='identity of this peer')
parser.add_argument('-k', type=int, default=5, help='number of results per query', metavar='N')
parser.add_argument('-q', '--quantize', action='store_true', help='enable quantization-aware training')
parser.add_argument('-s', '--simulation', action='store_true', help='perform simulation of user clicks on a set query')
args = parser.parse_args()
if args.k < 1: parser.error("The value of -k must be at least 1")

try:
    run(start_communities())
except KeyboardInterrupt:
    print('\nProgram terminated by user.')
    try:
        sys.exit(130)
    except SystemExit:
        os._exit(130)