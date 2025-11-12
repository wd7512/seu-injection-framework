## Future Versions

### v1.X.XX

Move many md files out of main and into a more development based branch. This will mean our future workflow stays clean on main and feature development can be performed on that branch. `dev` branch. Basically, we want a clear workflow and branch strategy. Having `main` and `dev` is good. And then `experimentation/...` off of `main`/`dev`?

We need to clean up old branches that may have some value in it. This is currently 
- [x] `ai_refactor`
- [ ] `Research/ViT`
- [ ] `Research/shipsnet`
- [ ] `Feature/forward_pass_states`

It would be nice to calculate the overhead caused by this simulation. We should calculate the average inference time of the model and compare this against the average injection time. If we know the overhead we can find ways to minimise it. 

### v1.2.0

We also want to branch off at some point and start experimentation. 

It is likely that features will be made during this experimentation and we can merge them back into main. When this happens we can move to 1.2.0 as it is proof that the package can be used to perform research

This move to v1.2.0 will also include testing on CUDA during experimentation. It would be useful to have a test that uses both CUDA and CPU and assess the performance. 

## Research Questions

**What is the robustness of other model types**
- Look at other type of NNs
- Look at SVM, DTR, Forest etc...

**What is the minimum norm required ($\delta_{min}$) for critical failure and how does this vary with model properties?**

- *Experimental setup*: can use a binary classifier, seu-injection is not strictly needed, can build a new injector based on $\delta_{\min} \in \mathbb{R}$
- Use mathematical analysis to come up with a hypothesis on how to make the model misclassify
- Vary model properties such as number of nodes and form relationships to the $\delta_{min}$
