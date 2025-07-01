---
title: "Beginning on the Leaderboard: My First Kaggle Challenge in 7 Days"
subtitle: "How I participated in the Image Matching Challenge with no prior Kaggle experience, just one week, and a lot of stubborn debugging."
date: 2025-06-30
author: "Vicki Mu"
tags: ["deep learning", "Kaggle", "computer vision"]
categories: ["Projects in the Wild"]
series: ["Projects in the Wild"]
series_order: 1
draft: false
description: "A walkthrough of how Vision Transformers work, how to implement one, and why it marked a major shift in computer vision."
editAppendPath: False
toc: true
---

> _Originally published on [Medium](https://medium.com/@vicki.y.mu/beginning-on-the-leaderboard-my-first-kaggle-challenge-in-7-days-0749ceebc825)._

## 1. Opening: Why I Joined So Late
{{< figure src="/img/kaggle_imc/imc2025.png" >}}

I first saw the Image Matching Challenge pop up on Kaggle sometime in early May. I bookmarked it out of curiosity — the problem sounded cool, and I’d been meaning to get into competitions for a while — but between MIT finals, end-of-semester deadlines, and packing up to move out, it got buried under everything else.

It wasn’t until a week before the deadline that I remembered it again. I was finally free, finally bored, and finally curious enough to open the starter notebook. One baseline submission later, I was hooked.

This was my first-ever Kaggle competition. I had no leaderboard expectations — just a vague goal to build something functional, learn as much as possible in seven days, and see how far I could push it.

> “It was a bit of a reckless decision. I had a week, a little familiarity with vision models, and zero competition experience. I wanted to see how far I could get.”

## 2. Understanding the Challenge

The Image Matching Challenge (IMC) is an annual computer vision competition hosted by CVPR and Kaggle. At its core, the goal sounds simple: match local features between image pairs to estimate the relative pose of the cameras that captured them. In other words, given two images of the same scene — possibly from wildly different angles or lighting — can you determine how those cameras were positioned relative to each other?

But 2025’s edition took that basic setup and made it *brutal*.

Unlike past IMC competitions, where the organizers provided pre-segmented scenes, **this year you had to segment the scenes yourself**. That meant reverse-engineering the scene structure from hundreds of loosely grouped image pairs before you could even begin estimating poses. It added an extra layer of uncertainty — you weren’t just solving the matching problem, you were solving the *context* problem, too.

{{< figure src="img/kaggle_imc/Different_Stairs_Diagram.png" caption="These are two completely different stairwells — but my pipeline lumped them into the same scene more than once. Turns out ‘gray steps in a hallway’ isn’t a very unique visual signature. Source: Kaggle IMC 2025" >}}

What really separated IMC 2025 from class projects or standard CV pipelines was the real-world messiness of it all:

- Images weren’t neatly aligned or staged — they were taken from inconsistent angles, sometimes under drastically different lighting conditions, or even at different times of day
- Generalization mattered more than precision — your pipeline had to work across architecture, nature, objects, stairs, ET statues, and more
- And finally, compute was a bottleneck: every submission was a tradeoff between quality and runtime, with timeout risks looming

## 3. Days 1–2: Getting the Baseline Running

I kicked things off with a pipeline based on **ALIKED + LightGlue** — a lightweight but effective combo I’d seen mentioned in previous IMC discussions. ALIKED gave me reliable keypoints and descriptors, and LightGlue handled the matching with impressive robustness, especially in low-texture scenes. It was a solid starting point — but far from plug-and-play.

The real curveball in IMC 2025 was **scene segmentation**. Unlike in previous years, the dataset didn’t come pre-organized into scenes. That meant before I could even estimate poses, I had to figure out which image pairs belonged to the same physical environment.

To tackle that, I used **DINOv2 as a global image descriptor**. For each image, I extracted global embeddings using a pretrained DINOv2 backbone, then clustered the embeddings using KMeans to group images into scenes. It wasn’t perfect — and I definitely tuned it more by vibe than theory — but it was good enough to avoid total chaos.

{{< figure src="/img/kaggle_imc/cluster.png" caption="Clustering results on the ET dataset using DINOv2 + KMeans. Two main clusters, with 3 outliers flagged. Source: author" >}}

Still, some scenes like **vineyard** and **stairs** were incredibly brittle. These environments had so much **repetition and structural symmetry** that even small mistakes in clustering led to **completely incorrect pose estimates**. Multiple rows of grapevines or similar staircases taken at slightly different angles or lighting conditions could easily be grouped together, even when they came from different physical spaces. Those mistakes hit hard — the mAA metric punishes wrong matches aggressively, and once scene segmentation goes wrong, everything downstream follows.

{{< figure src="/img/kaggle_imc/differentscenes.png" caption="Three different vineyard scenes — visually similar enough to confuse DINOv2 and cluster them together, despite coming from completely different locations. These kinds of mistakes wrecked downstream pose estimates. Source: Kaggle IMC 2025" >}}

Because I had joined the competition so late, I leaned heavily on the **Kaggle discussion boards** to stay afloat. Reading through others’ pain points and quick wins helped me shortcut a lot of early confusion — especially around runtime optimization and evaluating failure cases like ET, stairs, and the infamous castle towers. I also saw a few people visualizing keypoint matches and inlier ratios, which helped me trust (or distrust) my pipeline more quickly.

By the end of Day 2, I had something functional. Not amazing — but stable, reproducible, and ready to start iterating on. The real experimentation was still ahead.

## 4. Days 3–4: Idea Flood (and Reality Check)

By day three, I had a working baseline — which, naturally, meant I started throwing every half-baked idea I had at the wall to see what might stick. At one point, my notebook was a graveyard of TODOs: local SfM pipelines, GNN-based outlier rejection, even patch-wise attention reranking. Some of these I partially implemented. Others I just mocked up enough to convince myself I didn’t have the time.

But a few things *did* make it into the actual pipeline:

- **Cluster refinement via match graph connectivity**: I built a scene graph where each node was an image, and edges were weighted by LightGlue match confidence and number of inliers. Then I ran connected component analysis to recover tightly linked subgraphs — effectively letting the data self-organize the scene clusters. This was especially useful in ambiguous environments like vineyards and stairs, where DINOv2 global embeddings alone tended to over-group visually similar but spatially unrelated scenes.

- **Targeted COLMAP reruns with pre-filtered matches**: I selectively reran COLMAP on scenes with low estimated mAA, injecting LightGlue matches with custom outlier filtering. This worked well for scenes with harsh lighting changes or wide baselines — particularly the ET statue and several of the indoor staircases, where vanilla matching often produced degenerate poses.

- **Pose-aware pair filtering**: For a brief window, I experimented with using pairwise match consistency — essentially a simplified geometric verification — to discard scene pairs that weren’t consistent with the rest of their cluster. It was light-touch, but reduced false positives in scenes with high symmetry.

A few other ideas didn’t make the cut, but might in a future comp:

- Learning a **per-scene descriptor weighting** strategy (foreground vs. background emphasis)
- Filtering match graphs using **spectral clustering** on pose residuals
- Using **CLIP embeddings** as a second-pass sanity check for scene consistency

Ultimately, I realized I wasn’t going to outbuild the top teams — not in a week — but I could be thoughtful about what *not* to waste time on. That became my mini-strategy: solve what’s fragile, trust what’s robust, and don’t touch anything on Friday night.

## 5. Days 5–6: Patch, Tune, Pray

By this point, it wasn’t about adding features — it was about **pipeline triage**.

Days 5 and 6 were a blur of last-minute patches, scene-level debugging, and running the full dataset through my pipeline *without it catching fire*. I knew I was running out of time, and the priority shifted from “can I improve this?” to “can I trust this?”

{{< figure src="/img/kaggle_imc/prints.png" caption="The real MVPs were all the print statements. Source: author" >}}

I focused on final structure: stitching together my best scene clustering logic (DINOv2 + match-graph analysis), a couple custom COLMAP reruns for the weird scenes (looking at you, **ET**), and a filtering layer for pruning obviously bad pairs. **Stairs** remained one of the hardest — visually repetitive, sometimes indoor, sometimes outdoor, and prone to degeneracy in pose estimation. My only move there was to hand-check a few sample scenes and hope the outlier filter was doing its job.

Meanwhile, **compute became a bottleneck**. I was trying to test full runs locally, monitor GPU usage on Kaggle notebooks, and avoid timeout errors — all while juggling submission caps. At one point I split inference across multiple notebooks just to stay under the limit. It felt duct-taped together, but at least it was running.

> My notebook was a Frankenstein of late-night hacks, quick sanity checks, and a few very strategic `print()`s. Honestly, it had no right to work as well as it did.

## 6. Day 7: Submitting + The Final Rank

Submission day felt like holding my breath for 12 hours straight.

I queued up my final pipeline the night before the deadline — patched, filtered, clustered, and barely tested end-to-end. It wasn’t clean, and it definitely wasn’t elegant, but it ran. And at that point, that was all I needed.

When the results came in, I landed at **Rank 260 out of 943**.

{{< figure src="/img/kaggle_imc/leaderboard.png" caption="Final leaderboard rank: 260 out of 943 — not a medal, but a 501-place climb in seven days, and my first ever Kaggle comp. I’ll take it. Source: author" >}}

No medal, no spotlight — but for a **first competition** done in **seven days**, while figuring out scene segmentation, pose estimation, LightGlue, COLMAP, and Kaggle notebooks on the fly... I was proud of that number.

It wasn’t about beating anyone else. It was about proving to myself that I could build something from scratch, debug under pressure, and ship a real submission. And I did.

I didn’t expect a medal — but seeing my name halfway up the leaderboard felt like proof of concept: I could do this.

{{< figure src="/img/kaggle_imc/diagram.png" caption="What one week of late nights, weird bugs, and unexpected breakthroughs looked like. The timeline of how I (barely) held it all together. Source: author" >}}

## 7. What I Learned in 7 Days

This competition condensed months of learning into a single week. I came in with some computer vision experience — but matching, clustering, and pose estimation under pressure taught me things no lecture or tutorial ever could.

### **Reproducibility is non-negotiable**

When you’re iterating fast, the ability to rerun the exact same pipeline is everything. I lost hours to silent mismatches between intermediate files, cache conflicts, and half-tested logic blocks. By day 3, I was versioning configs and scene outputs like my life depended on it — because it kind of did.

### **Visual matching is way harder than it looks**

On paper, it’s just keypoints and descriptors. In reality, it’s everything from occlusions to lighting variation, repeating structures, degenerate geometry, and weird edge cases that no tutorial prepares you for. Robustness mattered more than elegance.

### **The leaderboard can help — or totally distract**

Seeing your name rise is addictive, but chasing short-term gains made me overfit to specific scenes more than once. Some of my lowest mAA scores came from over-optimized fixes that broke generalization. In the end, the most helpful metric was just **consistency across scene types**.

### **Competitions make you a faster, better debugger**

I had no choice but to figure things out quickly: mismatched pose outputs, LightGlue breaking silently, scenes clustering wrong, COLMAP throwing obscure errors... All of it sharpened my instincts. I now write defensive code by default — with checks, asserts, and visualizations built-in.

More than anything, I learned that these comps are a perfect training ground for research-style thinking: you pick your assumptions, fail fast, validate hard, and trust nothing until it works on the leaderboard.

## 8. What I’d Do Differently

Looking back, there’s a lot I’m proud of — but also a lot I’d change if I had more time (or a time machine).

### **Start earlier. Seriously.**

One week was barely enough to get a stable pipeline running, let alone explore the frontier of matching models. I spent more time fixing data issues than testing ideas — and missed out on the deep modeling work that could’ve made a real difference.

### **Get more aggressive with bad pairs**

Some of my lowest-scoring scenes were due to noisy matches that should’ve been filtered out earlier — either by geometric inconsistency, low keypoint density, or scene confusion. A stronger outlier rejection module (maybe even learned) would’ve saved me a lot of leaderboard pain.

### **Explore beyond the baseline**

I stuck with ALIKED + LightGlue for speed and familiarity, but next time I’d love to seriously try:

- **LoFTR** for its dense, global attention-driven matching  
- **GNN-based match filtering** to reason about match confidence beyond pairwise similarity  
- And maybe even **pose refinement modules** post-COLMAP, built to adapt to challenging camera configurations

If I’d had two more weeks, I would’ve focused on generalization: building a model-aware filtering stage and validating it across clusters, rather than hand-tuning scenes one-by-one. That’s the kind of robustness that separates mid-tier submissions from gold medals.

## 9. What’s Next

I’m not entirely sure what my next competition will be — but I know this sprint wasn’t a one-off.

I’m currently deciding between a few very different challenges:

- **CMI — Detect Behavior with Sensor Data**, to sharpen temporal modeling and classification robustness
- **NeurIPS — Open Polymer Prediction**, for a deep dive into scientific ML and graph-based modeling
- **Or DRW — Crypto Market Prediction**, a fast-paced signal extraction challenge that forces you to think like a quant

Whichever I choose, the goal is clear: **medal in my second comp**, and keep moving toward true mastery.

This sprint was a turning point. It reminded me why I do this — not just to rank, but to **build**, to **learn fast**, and to **get pushed beyond the comfort zone of lectures and controlled assignments**. I learned more in one week of hacking together a real pipeline than I have in some entire semesters. I learned what to prioritize, what to throw away, how to debug under pressure, and how to focus only on what matters when time is short.

Kaggle has become part of my larger AI journey — a place where I can stay sharp, stay uncomfortable, and build the kind of intuition that only comes from doing. It’s not always pretty. But it’s real.

* * *

## About Me

I’m a sophomore at MIT studying physics and artificial intelligence. This post is part of my *Projects in the Wild* series — where I document personal experiments, Kaggle challenges, physics-ML crossovers, and anything else that pushes me outside the classroom.

My goal is to learn by building: fast, messy, sometimes late, but always real. Whether it’s reimplementing papers or racing the Kaggle deadline, I use these projects to sharpen my instincts and stay uncomfortable — the kind of learning that doesn’t come from lectures alone.

If you enjoyed this post, feel free to [follow me on GitHub](https://github.com/vickiiimu), [Medium](https://medium.com/@vicki.y.mu), [Twitter](https://twitter.com/vickiiimu), or [reach out](mailto:vymu@mit.edu). I love talking about AI, open-source, and anything that blends math, vision, or modeling under pressure.