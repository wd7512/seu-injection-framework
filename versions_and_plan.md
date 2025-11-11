v1.1.XX

Move many md files out of main and into a more development based branch.

This will mean our future workflow stays clean on main and  feature development can be performed on that branch. `dev` branch.

Basically, we want a clear workflow and branch strategy. Having `main` and `dev` is good. And then `experimentation/...` off of `main`/`dev`?

We need to clean up old branches that may have some value in it. This is currently 
- [x] `ai_refactor`
- [ ] `Research/ViT`
- [ ] `Research/shipsnet`
- [ ] `Feature/forward_pass_states`

v1.2.0

We also want to branch off at some point and start experimentation. 

It is likely that features will be made during this experimentation and we can merge them back into main. When this happens we can move to 1.2.0 as it is proof that the package can be used to perform research

This move to v1.2.0 will also include testing on CUDA during experimentation. It would be useful to have a test that uses both CUDA and CPU and assess the performance. 
