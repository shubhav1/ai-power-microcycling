# ai training power microcycling (in progress!!)

**Question:** what shapes the training power waveform, and which decisions reduce grid-relevant swings at the lowest throughput/energy cost?

**Rationale:** data centers are popping up pretty much everywhere nowadays, with big bets on ai infra. there's a lot of push for BTM energy, but most of these data centers 
still rely on the grid for power. the problem with AI power demand is that it's very spike-y and requires huge influxes and drops, which a traditional grid is not built to
handle. there are a lot of solutions in development to address this, with innovations for better supercapacitors or BESS systems to buffer microcycles.

**Purpose:** as someone who's passionate about both infra and model architectures, i wanted to understand if there were also software/architectural decisions we could make 
to mitigate microcycling and alleviate pressure on the grid + communities relying on those grids. the first step is to actually map different model decisions to their 
respective energy usage patterns. this repository's purpose is to serve as an experiment that measures those differences exactly.
