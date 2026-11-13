# Modifications

Riblet extends the [pinned SPU/OpenBumbleBee sources](backend/sources.lock.json).
The backend changes register MJV-CT and public-site CDI in Cheetah and the PPHLO
executor, bind request-scoped preprocessing materials, and expose the public
linear audit result without revealing model arguments or logits.

The model adapters in `frontends/` build BERT and ViT graphs on those backends.
Their nonlinear shells adapt the activation-hook pattern in
[OpenBumbleBee's Flax example](https://github.com/AntCPLab/OpenBumbleBee/blob/47c5560d069543591f3b0176eb530ede28e33fc3/examples/python/ml/flax_bert/flax_bert.py).
`offline/` constructs numerical parameters on the model owner's host;
`src/riblet/` handles resource ownership, two-party execution and timing.

Modified upstream files carry a file-level notice. Original notices remain in
place. See [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).
