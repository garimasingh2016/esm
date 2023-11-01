import torch
import esm
from Bio.Seq import Seq
import time
import random

# length = []
# times = []
# bp = ["A", "T", "G", "C"]
# bp_seq = "ATGAGCAACACCTGCGACGAGAAGACCCAGAGCCTGGGCGTGAAGTTCCTGGACGAGTACCAGAGCAAGGTGAAGCGGCAGTACTTCAGCGGCTACCAG"

# while True:
#     print(f"predicting for length of {len(bp_seq)}")
#     # translate to protein
#     # use both fwd and rev sequences
#     dna_record = Seq(bp_seq)
#     dna_seqs = [dna_record, dna_record.reverse_complement()]

#     # generate all translation frames
#     data = [(f"translation {i}", str(s[i:].translate(to_stop=True))) for i in range(3) for s in dna_seqs]

#     model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#     batch_converter = alphabet.get_batch_converter()
#     model.eval()  # disables dropout for deterministic results

#     # Prepare data
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

#     # Extract per-residue representations (on CPU)
#     with torch.no_grad():
#         current = time.process_time()
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#         times.append(time.process_time()-current)
#     token_representations = results["representations"][33]

#     length.append(len(dna_record))
    
#     import csv

#     with open('results.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerows(zip(length, times))
    
#     # extend by 5 codons
#     for new_bp in random.choices(bp, k=15):
#         bp_seq += new_bp


# Load model directly
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

device = torch.device("cpu")
model.esm = model.esm.float()
model = model.to(device)
model.trunk.set_chunk_size(64)
experimental_sequence = ["ATGAGCAACACCTGCGACGAGAAGACCCAGAGCCTGGGCGTGAAGTTCCTGGACGAGTACCAGAGCAAGGTGAAGCGGCAGTACTTCAGCGGCTACCAGAGCGACATCGACACCCACAACCGGATCAAGGACGAGCTG"]

tokenized_input = tokenizer([experimental_sequence], return_tensors="pt", add_special_tokens=False)["input_ids"]
tokenized_input = tokenized_input.to(device)

with torch.no_grad():
    notebook_prediction = model.infer_pdb(experimental_sequence)
    with open("prediction.pdb", "w") as f:
        f.write(notebook_prediction)
