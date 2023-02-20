from dotenv import find_dotenv, load_dotenv
import torch

load_dotenv(find_dotenv())

losses = torch.load("models/losses_v1.pt")
print(losses)
