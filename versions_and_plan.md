v1.1.8

Move many md files out of main and into the `ai_refactor` branch.

This will mean our future workflow stays clean on main and big ai-assited feature development can be performed on that branch. We could name is `ai-dev` as well as having a `dev` branch.

We also want to branch off this version and start experimentation. It is likely that features will be made during this experimentation and we can merge them back into main. When this happens we can move to 1.2.0 as it is proof that the package can be used to perform research

This move to v1.2.0 will also include testing on CUDA during experimentation. It would be useful to have a test that uses both CUDA and CPU and assess the performance. 
