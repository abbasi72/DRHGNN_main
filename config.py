import argparse


def parse():
    p = argparse.ArgumentParser("DRHGNN: Drug Repurposing method based on Hypergraph Neural Network model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='Drug_Disease', help='data name')
    p.add_argument('--dataset', type=str, default='dis_dr_di_1', help='dataset name')
    p.add_argument('--model-name', type=str, default='UniSAGE', help='UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)')
    p.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')
    p.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')
    p.add_argument('--add-self-loop', action="store_true", help='add-self-loop to hypergraph')
    p.add_argument('--use-norm', action="store_true", help='use norm in the final layer')
    p.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')
    p.add_argument('--nlayer', type=int, default=8, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=8, help='number of hidden features, note that actually it\'s #nhid x #nhead')
    p.add_argument('--nhead', type=int, default=1, help='number of conv heads')
    p.add_argument('--dropout', type=float, default=0.6, help='dropout probability after UniConv layer')
    p.add_argument('--input-drop', type=float, default=0.6, help='dropout probability for input layer')
    p.add_argument('--attn-drop', type=float, default=0.6, help='dropout probability for attentions in UniGATConv')
    p.add_argument('--lr', type=float, default=0.001, help='learning rate')
    p.add_argument('--wd', type=float, default=4e-5, help='weight decay')
    p.add_argument('--epochs', type=int, default=350, help='number of epochs to train')
    p.add_argument('--n-runs', type=int, default=1, help='number of runs for repeated experiments(splits)')
    p.add_argument('--gpu', type=int, default=0, help='gpu id to use')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--patience', type=int, default=5, help='early stop after specific epochs')
    p.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')
    p.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')
    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
    return p.parse_args()
