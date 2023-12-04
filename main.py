from asyncio import run, sleep, create_task

from ipv8.taskmanager import TaskManager
from dataclasses import dataclass
import threading
from ipv8.community import Community, CommunitySettings
from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from ipv8.types import Peer
from time import time, localtime, strftime
from ipv8.peerdiscovery.network import PeerObserver

from sklearn.model_selection import train_test_split
from ipv8.util import run_forever
from ipv8_service import IPv8
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import random

# from simple_term_menu import TerminalMenu
from ltr import LTR
from utils import *
from vars import *

# Enhance normal dataclasses for IPv8 (see the serialization documentation)
dataclass = overwrite_dataclass(dataclass)



df = pd.read_csv('data/orcas.tsv', sep='\t', header=None,
                 names=['query_id', 'query', 'doc_id', 'doc'])
cnter = df.groupby('doc_id').count().sort_values('query',ascending = False) # get most referenced docse subset of most referenced docs
cnter = cnter[cnter['query']>1] # remove docs that are referenced only once to make stratify work
docs_to_be_used = cnter.sample(total_doc_count)
df = df[df['doc_id'].isin(list(docs_to_be_used.index))]


train_df, test_df = train_test_split(df, test_size=0.5, random_state=42, stratify=df['doc_id']) # split into train and test
train_df.to_csv('data/datasets/train_df.csv')
test_df.to_csv('data/datasets/test_df.csv')
df = train_df.copy()

# self.df = train_df.copy()
# self.df = self.df.sample(sample_nbr).dropna()
# docs = self.df[self.df['doc_id'].isin(df_data['doc_id'].unique())]





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

        number_of_docs_for_this_user = np.random.randint(number_of_docs_per_user[0], number_of_docs_per_user[1])
        self.df = df[df['doc_id'].isin(list(docs_to_be_used.sample(number_of_docs_for_this_user).index))]
        self.batches_so_far = 0
        self.current_queries = []
        self.current_docs = []
        self.timestamps = []
        self.accuracies_avg = 0
        self.accuracies_sum = 0
        self.rolling_window = 100

        task_manager = TaskManager()
        self.losses = []
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

    # def train_model(self, query, res):
        # self.got_here = False
        # # self.ltr.on_result_selected(query, res)
        # inputs = self.tokenizer([query], padding=True, return_tensors="pt").input_ids
        # labels = self.tokenizer([res], padding=True, return_tensors="pt").input_ids
        #
        # outputs = self.model(input_ids=inputs, labels=labels)
        # loss = outputs.loss
        # print (loss)
        # # Extract logits and convert to token IDs
        # logits = outputs.logits
        # predicted_token_ids = torch.argmax(logits, dim=-1)
        #
        # # Decode token IDs to text
        # predicted_text = self.tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        # if predicted_text == res:
        #     self.accuracy.append(1)
        # else:
        #     self.accuracy.append(0)
        #
        # loss.backward()
        # self.optimizer.step()
        # self.optimizer.zero_grad()
    def train_model(self, queries, responses):
        self.got_here = False


        self.batches_so_far+=1
        # Tokenize the lists of queries and responses
        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True).input_ids
        labels = self.tokenizer(responses, padding=True, return_tensors="pt", truncation=True).input_ids

        # Forward pass
        outputs = self.model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Extract logits and convert to token IDs
        logits = outputs.logits
        predicted_token_ids = torch.argmax(logits, dim=-1)

        self.accuracy = []

        # Decode token IDs to text for each item in the batch
        for i in range(predicted_token_ids.size(0)):
            predicted_text = self.tokenizer.decode(predicted_token_ids[i], skip_special_tokens=True)
            if predicted_text == responses[i]:
                self.accuracy.append(1)
            else:
                self.accuracy.append(0)

        acc = np.sum(self.accuracy) / len(self.accuracy)

        self.losses.append(round(float(loss.detach()),3))
        self.accuracies.append(acc)

        self.timestamps.append(int(time()))
        self.accuracies_sum += acc
        if len(self.accuracies) > self.rolling_window:
            self.accuracies_sum -= self.accuracies[-self.rolling_window]
            self.accuracies_avg = self.accuracies_sum / self.rolling_window
        else:
            self.accuracies_avg = self.accuracies_sum / len(self.accuracies)

        # if self.accuracies_avg >= accuracy_threshold:
        if self.batches_so_far % batches_save_threshold == 0:
            pd.DataFrame(list(zip(self.timestamps, self.accuracies)),
                         columns = ['timestamps','accuracies']).to_csv(f'data/accuracies/{self.my_peer.address[1]}_accuracies.csv')
            pd.DataFrame(list(zip(self.timestamps, self.losses)),
                         columns = ['timestamps','losses']).to_csv(f'data/losses/{self.my_peer.address[1]}_losses.csv')
            self.model.save_pretrained(f'data/models/{self.my_peer.address[1]}_{strftime("%Y-%m-%d %H%M%S", localtime())}_my_t5_model')
            self.df.to_csv(f'data/datasets/{self.my_peer.address[1]}_df.csv')
            # self.change_df(test_df)
            if self.accuracies_avg>0.9:
                raise SystemExit

        print(f'peer port:{self.my_peer.address[1]}, loss: {round(float(loss.detach()),3)}, ACCURACY ": {round(acc,2)}, '
              f'ACCURACY_AVG: {round(self.accuracies_avg,2)}')

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()





    def get_querynres(self):
        self.row = np.random.randint(0, self.df.shape[0])
        # print ('ROW NUMBER: ', self.row)
        query = self.df.iloc[self.row]['query']
        # selected_res = docs[docs == self.df.iloc[self.row]['doc_id']].index[0]
        selected_res = self.df.iloc[self.row]['doc_id']
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
        # self.ltr = LTR(quantize, self.df)

        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)

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



                p = random.choice(self.get_peers())
                self.ez_send(p, Query_res(query=query, result=selected_res))
            else:
                print('gonna try again')
                pass

        async def send_query() -> None:
            # self.train_model(payload.query, payload.result)
            if len(self.get_peers()) == 0:
                return None
            new_query, new_res = self.get_querynres()
            self.current_queries.append(new_query)
            self.current_docs.append(new_res)

            p = random.choice(self.get_peers())

            self.check_batch_size_and_train()
            # self.train_model(new_query, new_res)
            self.ez_send(p, Query_res(query=new_query, result=new_res))

        self.register_task("start_communication", start_communication, interval=5.0, delay=0)
        self.register_task("send_random_q_d_pair", send_query, interval=0.01, delay=0)
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
        #         query = self.df.iloc[row]['query']
        #         selected_res = docs[docs==self.df.iloc[row]['doc_id']].index[0]
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

    def check_batch_size_and_train(self):
        if len(self.current_queries)>=batch_size:
            self.train_model(self.current_queries, self.current_docs)
            self.current_queries = []
            self.current_docs = []

    def change_df(self, df_changer):
            self.df = df_changer.copy()
            self.got_here = True

    @lazy_wrapper(Query_res)
    def on_message(self, peer: Peer, payload: Query_res) -> None:
        # print(self.my_peer, 'received:', payload.query, payload.result)

        # if self.row >= self.df.shape[0]:
        #     self.row = 0
        #     self.accuracies.append(self.accuracy)
            # if acc >= 0.9:
            #     if self.got_here:
            #         raise SystemExit
            #     pd.DataFrame([np.sum(i)/self.df.shape[0] for i in self.accuracies]).to_csv('data/accuracies.csv')
                # raise SystemExit
                # self.change_df(test_df)


        self.current_queries.append(payload.query)
        self.current_docs.append(payload.result)
        self.check_batch_size_and_train()



    # @lazy_wrapper(MyMessage)
    # def on_message(self, peer: Peer, payload: MyMessage) -> None:
    #     # Update our Lamport clock.
    #     self.lamport_clock = max(self.lamport_clock, payload.clock) + 1
    #     # print(self.my_peer, "current clock:", self.lamport_clock)
    #     # print(self.my_peer, 'received:', payload.second_string)
    #     # Then synchronize with the rest of the network again.
    #     self.ez_send(peer, MyMessage(self.lamport_clock, second_string='second hello'))


async def start_communities() -> None:
    for i in range(peer_nbr):
        builder = ConfigBuilder().clear_keys().clear_overlays()
        builder.add_key("my peer", "medium", f"certs/ec{i}.pem")
        builder.add_overlay("LTRCommunity", "my peer",
                            [WalkerDefinition(Strategy.RandomWalk, 10, {'timeout': 3.0})],
                            default_bootstrap_defs, {}, [('started',)])
        await IPv8(builder.finalize(),
                   extra_communities={'LTRCommunity': LTRCommunity}).start()
    await run_forever()


run(start_communities())
