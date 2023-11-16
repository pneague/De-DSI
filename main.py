from asyncio import run, sleep, create_task
from dataclasses import dataclass
import threading
from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from ipv8.types import Peer
from ipv8.peerdiscovery.network import PeerObserver

from sklearn.model_selection import train_test_split
from ipv8.util import run_forever
from ipv8_service import IPv8
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

# from simple_term_menu import TerminalMenu
from ltr import LTR
from utils import *
from vars import id1, id2, quantize, sample_nbr

# Enhance normal dataclasses for IPv8 (see the serialization documentation)
dataclass = overwrite_dataclass(dataclass)

df = pd.read_csv('data/orcas.tsv', sep='\t', header=None,
                 names=['query_id', 'query', 'doc_id', 'doc'])
cnter = df.groupby('doc_id').count().sort_values('query',ascending = False) # get most referenced docs
df_data = df[df['doc_id'].isin(list(cnter.iloc[1000:1100].index))] #take subset of most referenced docs
train_df, test_df = train_test_split(df_data, test_size=0.5, random_state=42) # split into train and test
df = train_df.copy()
# df = df.sample(sample_nbr).dropna()
# separate the doc_id into a string with spaces separating the words
# df['doc_id'] = df['doc_id'].apply(lambda x: ' '.join(x))
docs = df[df['doc_id'].isin(df_data['doc_id'].unique())]





@dataclass(msg_id=1)  # The value 1 identifies this message and must be unique per community
class Query_res:
    query: str
    result: str




# @dataclass(msg_id=2)  # The value 1 identifies this message and must be unique per community
# class MyMessage:
#     clock: int  # We add an integer (technically a " long long") field "clock" to this message
#     second_string: str


class LTRCommunity(Community):
    community_id = b'\x9d\x10\xaa\x8c\xfa\x0b\x19\xee\x96\x8d\xf4\x91\xea\xdc\xcb\x94\xa7\x1d\x8b\x00'

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.row = 0
        self.cycle = 0
        self.accuracies = []
        self.accuracy = []
        if quantize:
            self.community_id = self.community_id[:-1] + bytes([0x01])
        # self.add_message_handler(UpdateModel, self.on_message)
        # self.packets: list[UpdateModel] = []
        # self.input_queue = queue.Queue()
        self.add_message_handler(Query_res, self.on_message)
        # self.add_message_handler(MyMessage, self.on_message)
        self.ready_for_input = threading.Event()
        self.lamport_clock = 0
        self.past_data = {'queries': [], 'results': []}

    def on_peer_added(self, peer: Peer) -> None:
        print("I am:", self.my_peer, "I found:", peer)

    def train_model(self, query, res):
        self.got_here = False
        # self.ltr.on_result_selected(query, res)
        inputs = self.tokenizer([query], padding=True, return_tensors="pt").input_ids
        labels = self.tokenizer([res], padding=True, return_tensors="pt").input_ids

        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        print (loss)
        # Extract logits and convert to token IDs
        logits = outputs.logits
        predicted_token_ids = torch.argmax(logits, dim=-1)

        # Decode token IDs to text
        predicted_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        if predicted_text == res:
            self.accuracy.append(1)
        else:
            self.accuracy.append(0)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_querynres(self):
        # print ('ROW NUMBER: ', self.row)
        query = df.iloc[self.row]['query']
        # selected_res = docs[docs == df.iloc[self.row]['doc_id']].index[0]
        selected_res = df.iloc[self.row]['doc_id']
        self.row += 1
        return query, selected_res

    # def input_thread(self):
    #     while True:
    #         self.ready_for_input.wait()
    #         query = input(f"\r{fmt('QUERY', 'purple')}: ")
    #         self.input_queue.put(query)
    #         self.ready_for_input.clear()

    def started(self) -> None:
        print('Indexing (please wait)...')
        # self.ltr = LTR(quantize, df)

        # Load model and tokenizer
        model_name = "t5-small"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Training loop
        self.model.train()

        self.network.add_peer_observer(self)
        print(self.get_peers())


        async def start_communication() -> None:
            print(f'running comms routine with {len(self.get_peers())} peers')
            if len(self.get_peers()) > 0:
                # if not self.lamport_clock:
                query, selected_res = self.get_querynres()
                print('starting comms & ending comms')
                self.cancel_pending_task("start_communication")
                for p in self.get_peers():
                    self.ez_send(p, Query_res(query=query, result=selected_res))
            else:
                # self.cancel_pending_task("start_communication")
                print('gonna try again')
                pass

        self.register_task("start_communication", start_communication, interval=5.0, delay=0)
        # async def app() -> None:
        #     print ("NUMBER OF PEERS FOUND:", len(self.get_peers()) )
        #     # threading.Thread(daemon=True).start()
        #
        #     while True:
        #         if row >= 50 * (len(self.get_peers()) + 1):
        #             break
        #         # self.ready_for_input.set()
        #         # while self.input_queue.empty():
        #         #     await sleep(0.1)
        #         print (row)
        #         query = df.iloc[row]['query']
        #         selected_res = docs[docs==df.iloc[row]['doc_id']].index[0]
        #         print (query, selected_res)
        #         row += 1
        #         if selected_res is None: continue
        #         print(f"{fmt('RESULT', 'blue')}:", selected_res)
        #         await sleep(0)
        #
        #         self.ltr.on_result_selected(query, selected_res)
        #         model_bf = self.ltr.serialize_model()
        #         chunks = split(model_bf, 8192)
        #         for peer in self.get_peers():
        #             _id = os.urandom(16)
        #             print(fmt(f'Sending update (packet 0/{len(chunks)})', 'gray'), end='')
        #             for i, chunk in enumerate(chunks):
        #                 print(fmt(f'\rSending update (packet {i+1}/{len(chunks)})', 'gray'), end='')
        #                 self.ez_send(peer, UpdateModel(_id, i+1, len(chunks), chunk))
        #                 time.sleep(0.01)
        #             print()

    # @lazy_wrapper(UpdateModel)
    # def on_message(self, peer: Peer, payload: UpdateModel) -> None:
    #     self.packets.append(payload)
    #     packets = [x for x in self.packets if x.id == payload.id]
    #     if len(packets) == payload.total:
    #         model_bf = io.BytesIO()
    #         for packet in sorted(packets, key=lambda x: x.fragment):
    #             model_bf.write(packet.data)
    #         self.packets = list(filter(lambda x: x.id != payload.id, self.packets))
    #         model_bf.seek(0)
    #         model = torch.load(model_bf)
    #         self.ltr.apply_updates(model)

    def change_df(self, df_changer):
            global df
            df = df_changer.copy()
            self.got_here = True

    @lazy_wrapper(Query_res)
    def on_message(self, peer: Peer, payload: Query_res) -> None:
        # print(self.my_peer, 'received:', payload.query, payload.result)

        if self.row >= df.shape[0]:
            self.row = 0
            self.cycle += 1
            self.accuracies.append(self.accuracy)
            acc = np.sum(self.accuracy)/len(self.accuracy)
            print (f'ACCURACY ON CYCLE {self.cycle}": {acc}')
            print ('-----------------------------------------------------------------------------------')
            # if acc == 1 or self.cycle > 200:
            if acc >= 0.9:
                if self.got_here:
                    raise SystemExit
                pd.DataFrame([np.sum(i)/df.shape[0] for i in self.accuracies]).to_csv('data/accuracies.csv')
                # raise SystemExit
                self.change_df(test_df)

            self.accuracy = []


        self.train_model(payload.query, payload.result)
        new_query, new_res = self.get_querynres()

        # self.train_model(new_query, new_res)
        self.ez_send(peer, Query_res(query=new_query, result=new_res))

    # @lazy_wrapper(MyMessage)
    # def on_message(self, peer: Peer, payload: MyMessage) -> None:
    #     # Update our Lamport clock.
    #     self.lamport_clock = max(self.lamport_clock, payload.clock) + 1
    #     # print(self.my_peer, "current clock:", self.lamport_clock)
    #     # print(self.my_peer, 'received:', payload.second_string)
    #     # Then synchronize with the rest of the network again.
    #     self.ez_send(peer, MyMessage(self.lamport_clock, second_string='second hello'))


async def start_communities() -> None:
    for i in [1, 2]:
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "medium", f"certs/ec{i}.pem")
        builder.add_overlay("LTRCommunity", "my peer",
                            [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        await IPv8(builder.finalize(),
                   extra_communities={'LTRCommunity': LTRCommunity}).start()
    await run_forever()


run(start_communities())
