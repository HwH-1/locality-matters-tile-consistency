# Locality Matters: Tile-Consistent Content-Adaptive Anisotropic Kuwahara Filtering for Large-Canvas Abstraction

YingJie Wang\*

\*Handan College, Handan, Hebei, China. Corresponding author. E-mail: 717646407@qq.com. ORCID: 0009-0005-5898-0448

**Keywords.** non-photorealistic rendering · image abstraction · Kuwahara filter · anisotropic filtering · structure tensor · tiled and out-of-core processing · reproducibility

# Abstract

A filter whose parameter is computed from the whole image produces a different image when it is evaluated in tiles. We prove the condition behind this: if any quantity that can change a pixel's output is a non-degenerate whole-image functional, **no finite overlap halo**, and no single-pass streaming or out-of-core pass, reproduces the whole-image result. That separates the failure from support truncation, which a halo does fix, and yields a metric (TCE) that established quality measures do not respond to. The practical cost is metrological. Changing the evaluation harness moves a global-quantile variant by up to $0.031$ SR and $0.019$ SSIM, comparable to the largest quality gain we report, while the bounded-support LRS-AKF is unchanged at four decimals; four-method rankings invert in 15 of 40 cases. LRS-AKF is exactly tile-consistent on 141 of 144 benchmark images. The predicate also audits code we did not write: of seven published OpenCV 5.0.0 algorithms with bounds registered in advance, three are bit-exact inside their bound and two exceed it. An audit of four public anisotropic Kuwahara implementations found no whole-image reduction on their main support-selection path, so we state the result as a correctness criterion and a measurement protocol, not as a census of deployed practice.

# 1 Introduction

The anisotropic Kuwahara filter and its descendants produce painterly abstraction by comparing the variances of oriented sectors around each pixel and selecting the most homogeneous one [@kyprianidis2009akf; @kyprianidis2011msakf]. Because a fixed support is wrong in both flat and detailed regions, a substantial literature adapts the support: adaptive window selection [@bartyzel2015adaptive; @arya2024adaptive], multi-scale variants [@kyprianidis2011msakf], depth- or focus-driven scale [@ahn2010depth], and coherence- or structure-guided adaptivity [@zang2015guided]; the operator is still carried as a working tool outside graphics, half a century after the original [@schuster2025kuwahara]. Every one of these methods makes its decision from a *local* statistic. That statistic can nevertheless be normalised against an image-wide percentile before it is thresholded, which turns a local decision into a whole-image one. We do not claim this is common practice. Section 6.10 audits four publicly available anisotropic Kuwahara implementations and finds no whole-image reduction on the main support-selection path of any of them. The pattern does occur in the two reference implementations we built, in the per-tile feature statistics of learned pipelines [@buglakova2025tiling], and in one published structure-repair branch whose mask threshold is an image-wide percentile (Section 6.10). We therefore present it as a *failure mode with a proof and a test*, not as a census of deployed code. What has not been stated is when such a substitution is exact, and what it costs when it is not.

This paper is about that normalisation step, which for this operator class is stated nowhere and tested nowhere. Concretely:

> If a filter's parameter is a functional of the whole image, then the filter's output for a region depends on the pixels outside that region. Processing the image in tiles, which is *mandatory* whenever the image does not fit in memory [@dash2022gigapixel; @powell2010gigapixel], and standard practice in every streaming and parallel pipeline, therefore produces a **different image**, not merely a lower-quality one.

The community already knows about the benign half of this problem. Tiled/overlap-tile evaluation is used precisely to obtain *seamless* results for arbitrarily large inputs [@ronneberger2015unet], and where residual tiling effects have been studied, they are attributed to translation variance of strided or pooled operators and mitigated by using larger tiles [@reina2020tiling; @huang2018tiling]. That story is about **support truncation**: a pixel near a tile edge lacks context. It is fixed exactly by an overlap halo at least as wide as the operator's support, and we confirm this experimentally: a fixed-radius anisotropic Kuwahara filter becomes *exactly* reproducible under tiling (by the float-domain criterion of Section 4.1) once the halo reaches its support.

The other half has no name in this literature and no halo fixes it. **Statistic non-locality** cannot be removed by any finite halo, because the quantity that sets the parameter is computed from the entire image and changes from tile to tile. The repair that does work is to change the statistic, which is what adaptive histogram equalisation did for contrast enhancement when it replaced the global histogram with block-wise histograms and interpolation [@pizer1987clahe]; what is missing is a criterion for when the localised operator reproduces the whole-image one, and a measurement of what the global version costs when it does not. We supply both, and we measure the consequence directly: the standard repair ("increase the overlap") buys nothing beyond a plateau, and the plateau is visible: 6.5 % of pixels differ by more than one 8-bit level with a 32-pixel halo, and the deviation lives *in the interiors of tiles*, not on the seams (Sections 6.2 and 6.3).

A second consequence is aimed at readers as *authors*: because the mismatch is invisible to the metrics in current use, **evaluation protocols that differ in how the image is fed to the filter rank methods differently**. Comparing four filters under whole-image and tile-wise evaluation on the same images, we observe 15 rank inversions in 40 image–metric cases. A method can win a published comparison and lose in the pipeline that actually ships.

We close the gap rather than only reporting it. Section 3 gives a sufficient condition for exact tile consistency (bounded support of the *entire* parameter chain); Section 5 verifies that our new operator LRS-AKF satisfies it and shows that it costs nothing in output quality: on the same images the tiled-exact operator is significantly better on SSIM and gradient-coherence retention and significantly worse on one flatness metric, with the other three metrics not separable at $n=10$ (Section 6.4).

**Contributions.**

1. **A taxonomy and a property, with its limits measured.** We distinguish class-A (support truncation) from class-B (statistic non-locality) tile mismatch, formalise *tile consistency*, and prove a sufficient condition for exactness (Proposition 1) together with the complementary observation that a whole-image functional violates it for every halo (Proposition 2). We then test the sharp reading of Proposition 1 rather than trusting it: for the fixed-radius control the often-quoted support $2r+\rho$ understates the halo at which exactness appears (21/144 images match, 123/144 need more, never less), because the implementation still contains unbounded Gaussian smoothing. Every exact halo we report in this paper is therefore *measured*, not predicted (Section 6.8.3). The taxonomy is then confirmed on **other people's code**: seven published algorithms shipped in OpenCV 5.0.0 run under the same protocol, each with a support bound registered in advance, and they separate as the taxonomy predicts, including a third failure mode outside it in which Perona–Malik diffusion is not a function of pixel content at all (Section 6.9).
2. **A measurement protocol, validated at three scales.** Tile-consistency error (TCE), its border-band decomposition, grid-line visibility (GV), a rank-inversion test, and (where no whole-image reference exists) a Cauchy-criterion check (Section 6.7.1). Six established quality metrics are blind to a mismatch that moves 6.5–20.5 % of pixels by more than one 8-bit level; the battery is run at $n=10$, the central claim re-measured at $N=144$ images across three content types (Section 6.8), and both reproduced on a single native-resolution artwork where the whole-canvas reference cannot be executed at all (Section 6.7.1) and on seven third-party implementations (Section 6.9).
3. **An operator.** LRS-AKF: a content-adaptive anisotropic Kuwahara filter whose scale criterion is a bounded-support local reference. It is exactly tile-consistent on 110/112 museum records at a 32-pixel halo (141/144 overall; the three exceptions are one pixel each at 1.2–3.3 levels), stays exact at $2048\times2048$ (Section 6.7), and at native $4000\times3853$ it is the only pass that runs at all, because the whole-canvas reference cannot be executed on an 8 GB host (Section 6.7.1). Its reference is a separable bounded-support scan, so a streaming implementation needs $\rho$ rows of state rather than the image (Section 5.2). It is quality-competitive: significantly better than the global-quantile reference on SSIM and SCG after multiplicity correction, significantly worse on FSR, indistinguishable elsewhere (Section 6.4).
4. **Portability of the reported numbers, shown rather than assumed.** Every method was evaluated under both harnesses on the same images. The global-quantile variants' quality figures move when the evaluation is tiled (SC-AKF: $-0.031$ SR, $-0.019$ SSIM, and its whole-image SSIM edge over CA-AKF flips sign), while the bounded-support operators move by zero to the four decimals we report on all six metrics (largest residual $1.2\times10^{-4}$; Table 7). A correctness property that makes a published number survive a change of harness is the part of this contribution most useful to other authors.
5. **A deployment budget, and one claim we tested and rejected.** We bound the cost of tiled evaluation by $W_{\mathrm{tiled}}\le(1+2h/T)^2W_{\mathrm{whole}}+Nc_0$ with a fitted per-tile term, measured 1.36–2.23× in time against 4.9–8.9× in peak memory, and give the matching memory bound (Section 3.2, Propositions 3–4). The obvious-looking strengthening, that the decision statistic is low-rank, so the state can be factorised and tiling made cheaper, is *false on our own measurements*: the $2\times2$ second-moment matrix is rank-bounded by the domain dimension (Property 1) but exactly rank-deficient on 0.0 % of pixels and has rank-1 truncation error above half the spectrum on 17–71 % of them (Section 3.1). We publish that negative result because it closes off a shortcut the field would otherwise take.
6. **A design rule with a named analogy, not a new tuning knob.** Section 8 states the locality requirement for any operator whose parameters come from image-wide statistics, percentile-normalised edge energy, auto-levels, global histogram matching, learned parameter predictors, and notes the structural parallel to train/test statistics leakage and per-shard normalisation in distributed learning, as an analogy rather than as evidence.

**Relation to prior work in one sentence, and why it is not the same thing.** Adaptive-window Kuwahara papers ask *which window size is best* [@bartyzel2015adaptive; @arya2024adaptive]; multi-scale and depth-driven variants ask *which scale to use where* [@kyprianidis2011msakf; @ahn2010depth]; the tiling literature asks *how to avoid seams when a big image is cut up* [@ronneberger2015unet; @reina2020tiling; @huang2018tiling]. None asks whether **the parameter itself is computable from the region it is applied to**, which is the question that decides whether a filter can be used on a large canvas at all, and none provides a criterion under which the answer can be verified rather than hoped for.

# 2 Related work

## 2.1 Anisotropic Kuwahara filtering and adaptive support

The Kuwahara operator replaces a pixel by the mean of the lowest-variance quadrant of its neighbourhood [@kuwahara1976]. Its anisotropic extension orients the sectors by the structure tensor and blends them softly, producing painterly abstraction with directional brush structure [@kyprianidis2009akf]; the multi-scale formulation integrates several radii into a pyramid [@kyprianidis2011msakf]. Adaptivity of the support itself has been pursued directly, by an adaptive Kuwahara filter that determines the window size per pixel [@bartyzel2015adaptive] and by a recent variant targeting image enhancement [@arya2024adaptive], and indirectly, through scale selection guided by directional anisotropic structure measurements [@zang2015guided] or by depth [@ahn2010depth]. The same instinct is alive in adjacent venues: adaptive anisotropic morphological filtering keyed to the co-circularity of local orientations [@blusseau2022coirc], structure-preserving smoothing from the same community [@zhu2023tvcsmooth], and anisotropic diffusion still supplying a reproducible super-resolution pipeline in 2026 [@monasse2026adsr].

**Delta.** All of these are statements about *what the parameter should be*. They are, with the exceptions noted below, silent about *where the parameter may be computed from*, and the implementations we audited do normalise by image-wide quantities: our own reference implementation CA-AKF divides the structure-tensor major eigenvalue by its 95th percentile over the image, a construction that follows standard practice in this literature rather than departing from it, and the variance-quantile criterion in Section 5.1 is the natural way of expressing "how much texture is a lot, in this image". We do **not** claim to have audited every paper in this lane: the internals of @bartyzel2015adaptive and @arya2024adaptive are described in their own words as locally-determined window sizes, and we cite them as the closest *family* of methods, not as instances of the failure we measure. The claim we make is narrower and checkable: an adaptive-support Kuwahara filter can be built so that its parameter is a functional of the whole image; this construction is in active use, it is not flagged as such in the papers that use it, and it breaks tiled deployment (Sections 5.1, 6.1, 6.3). Repairing it does not change the parameter-selection semantics, so the fix is orthogonal to this literature rather than competing with it.

## 2.2 Tiled and out-of-core image processing

Tiling is the standard answer to "the image does not fit in memory": gigapixel pipelines for whole-slide imaging and planetary imagery process tiles and reassemble them [@dash2022gigapixel; @powell2010gigapixel]. In learned segmentation, the overlap-tile strategy was introduced specifically to obtain seamless predictions for arbitrarily large inputs by mirroring tile borders and keeping only the reliable interior [@ronneberger2015unet]. The residual effects of tiling were studied systematically by @reina2020tiling, who attribute prediction differences to translation variance of pooling and strided convolutions and recommend larger tiles; @huang2018tiling document stitching challenges in remote-sensing segmentation. Closest to our class-B mechanism, @buglakova2025tiling trace tiling artefacts in sliding-window inference to *feature normalisation layers whose statistics are computed per tile*, show that the artefacts persist even when the classical receptive-field condition is satisfied, and fix them by replacing tile-wise normalisation with a normalisation whose test-time statistics are not tile-dependent. The lane is still active in 2026: @kwark2026seamless pursue "seamless whole slide" processing by construction, and @zhang2026ltoio propose selecting the tile-overlap configuration *per boundary* from a predictor of downstream seam quality, explicitly framing overlap choice as a heuristic search over visible-seam risk (an unreviewed preprint whose own abstract states that no experimental results are claimed; cited here only to locate the nearest concurrent work).

Across this lane overlap is treated as a tunable, never as a theorem. Tile size and a 12.5 % overlap are studied as statistical factors whose effect on road-classification accuracy is attributed to “reduced semantic context or the lack of information near the image boundaries” [@cira2024tile]; a 2025 study of slicing-aided hyper inference states that “Overlapping tiles can mitigate certain boundary effects but often result in duplicate detections and boundary inconsistencies, particularly along patch edges” [@mihajlovic2025sahi]; and the slicing recipe itself [@akyon2022sahi] is presented as a configuration to be chosen, not a property to be proved. All three are class-A diagnoses in our terms (context missing at a border, repaired by more context) and none of them asks at what halo the tiled result *equals* the whole-image one.

**Delta.** In most of this literature the cause is the operator's *receptive field* (class A in our terminology) and the remedy is "more context": bigger tiles, wider haloes, per-boundary overlap search, or blending windows. Where the statistic-computation cause is identified at all (@buglakova2025tiling in learned pipelines) the response is an empirical swap of one normalisation layer for another, with no statement of *when* a tiled evaluation equals the whole-image evaluation and no error measure that would let anyone check. We prove and measure the complementary case for classical filters. For content-adaptive filters whose parameter is a whole-image functional, **more context does not converge to the right answer**. The mismatch plateaus at a value essentially independent of halo size (TCE $2.25\times10^{-3}\to1.99\times10^{-3}$ from halo 16 → 32 px: doubling the halo buys a 12 % reduction in error and leaves 6.5 % of pixels differing by more than one display level).

**The residual is interior, not at the seam.** The residual lives in tile *interiors*, where no seam-based remedy (blending, per-boundary overlap, or seamline optimisation) can reach it (Section 6.2). The prescription "use the largest tile that fits" is therefore not merely inefficient for this filter class; it is *not a solution*, and a paper that reports "seamless" tiled output has not established correctness. Conversely, our operator makes the whole question finite and provable: exactness at a fixed halo of 24–32 px that we predict from the support ($B+\rho$) and verify per image, so the search over overlap configurations that @zhang2026ltoio proposes is unnecessary here and, for class-B operators, unwinnable.

**Four precedents from adjacent fields, and what each leaves open.** The idea of replacing a global statistic with a local one is older than this paper. Adaptive histogram equalisation computes its transfer function from block-wise histograms and interpolates between block centres [@pizer1987clahe], which is the design rule of Section 8 applied to contrast enhancement; what it does not provide is a statement of when a partition reproduces the untiled result, and its block grid is part of the operator, so a different partition defines a different operator. In wavelet coding, tile-boundary artefacts are the canonical class-A case: point-symmetric extension and odd tile sizes remove them by giving each tile the context its transform needs [@kharitonenko2002wavelet; @wei2005tile], again without an exactness criterion that survives a change of partition. Exact tile-based inference makes the class-A condition explicit for bounded-receptive-field networks and derives the required halo from the network's receptive field [@majurski2021exact], the same condition as Proposition 1, for a different operator class.

**The one exception.** Streaming multigrid keeps a global coupling and still fits in memory: it solves the same gradient-domain system in two streaming passes over out-of-core data [@kazhdan2008smg]. That is a solver-level fix rather than a parameter-level one, and it does not claim that independent tiles reproduce the global solution.

Table 1 sets the five closest cases side by side, on the axis that decides whether a tiled result can be trusted.

**Table 1.** The nearest tiling-consistency work, compared on five axes: the setting, whether a *cause* of the residual difference is named, the remedy proposed, whether the work states when tiled and whole-image evaluation agree exactly, and whether the deviation is measured.

| work | setting | cause named | remedy | exactness stated | deviation measured |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| Reina et al. [@reina2020tiling] | CNN segmentation | pooling / strided-conv translation variance | larger tiles | no | downstream accuracy |
| Buglakova et al. [@buglakova2025tiling] | biological-image segmentation | per-tile normalisation statistics | tile-independent normalisation | no | artefact visibility, accuracy |
| Zhang [@zhang2026ltoio] | diffusion-model stitching (preprint) | seam quality per boundary | per-boundary overlap search | no | predicted seam score |
| Majurski & Bajcsy [@majurski2021exact] | tile-based CNN inference | receptive field (class A) | halo sized from the receptive field | yes, for a bounded receptive field | tiled output equals whole-image output |
| this work | classical adaptive filters | parameter is a whole-image functional | bounded-support parameter chain | yes (Props. 1–2) | yes (TCE, bands, GV, rank flips) |

## 2.3 Global statistics in image operators

Percentile- and quantile-based normalisation appears in several distinct settings: contrast normalisation by image quantiles, auto-levels, histogram matching, and (in learned pipelines) instance and normalisation statistics. Section 6.10 reports how often it appears in the adaptive-Kuwahara code we could audit, and the answer is not often. The classical attempt to localise this family is adaptive histogram equalisation, which estimates the transfer function per block and interpolates between block centres [@pizer1987clahe]; it makes the statistic local, but the block grid becomes part of the operator, and no exactness condition is stated for a different partition. The learned pipelines that do this deliberately are worth naming, because they show the mechanism is a design choice rather than an oversight: ultra-resolution style transfer computes InstanceNorm statistics on a *thumbnail of the whole image* and applies them to every patch precisely so that the patches stay mutually consistent, which is how it reaches 10000×10000 canvases [@chen2022urst]. That literature reports that the swap works; it does not state when the tiled result equals the untiled one, or what overlap would be needed for that. In data mining, the analogous hazard is known as *leakage*: computing a preprocessing statistic on the whole dataset makes an example's transformed value depend on other examples. Streaming quantile estimators such as the Greenwald–Khanna summary [@greenwald2001gk] make global statistics approximable in bounded memory, but they do not make them *local*: an example's value still depends on data it has not seen, and a tile's value still depends on other tiles.

The non-learned operators audited in Section 6.9 are drawn from the classical shelf rather than from our own code: bilateral filtering [@tomasi1998bilateral], the guided filter and its accelerated variant [@he2010guided; @he2013guidedtpami; @he2015fastguided], non-local means in its published forms including the video-sequence variant and the IPOL re-publication [@buades2005nlm; @buades2005avss; @buades2011ipolnlm], anisotropic diffusion [@perona1991diffusion], adaptive manifold filtering [@gastal2012amf], rolling guidance [@zhang2014rollingguidance] and L0 image smoothing [@xu2011l0smooth], each as shipped in OpenCV 5.0.0 and opencv_contrib 5.0.0. A taxonomy that only classified our own operators would describe one codebase rather than the field, so the support bound used for each method was fixed from its published construction and recorded before its halo ladder was run, not fitted afterwards.

**Delta.** We do not propose a better quantile estimator; we propose removing the quantile from the parameter chain entirely, replacing it with a bounded-support local statistic that is (i) computable per tile, (ii) dimensionless, and (iii) exactly consistent under partitioning.

# 3 Preliminaries: locality and tile consistency

Let $I\in\mathbb R^{H\times W\times 3}$ be an image and $f$ an image operator with output $f(I)$. Let $\mathcal P_n$ be the uniform $n\times n$ partition of the image into tiles, and let $f_{\mathcal P,h}$ denote *tile-wise evaluation with overlap halo $h$*: for each tile $T$, the operator is applied to $T$ extended by $h$ pixels, and only the pixels of $T$ are kept.

**Definition 1 (tile consistency).** $f$ is tile-consistent if for every image $I$ there exists $h^{*}(I)$ such that for all $n$ and all $h\ge h^{*}(I)$, $f_{\mathcal P_n,h}(I)=f(I)$ exactly.

**Definition 2 (locality).** $f$ is local with support $B$ if the output at pixel $p$ depends only on $I$ restricted to the ball of radius $B$ around $p$.

**Proposition 1 (sufficient condition).** A local operator with support $B$ is tile-consistent with $h^{*}\le B$, provided border handling is consistent (e.g. reflect at the image boundary).

*Proof sketch.* For an interior tile, the halo-$h$ neighbourhood of any pixel of the tile with $h\ge B$ contains the pixel's full $B$-ball, which is identical to the one available in the whole-image evaluation; the outputs therefore coincide. Pixels within $B$ of the image border are handled identically because the halo is clipped there in both evaluations. $\square$

**Definition 3 (non-degenerate whole-image functional).** A functional $S$ is *non-degenerate for* $f$ if (i) no bounded neighbourhood determines it: for every radius $R$ and every pixel $p$ there are images $I,I'$ that agree on the ball of radius $R$ about $p$ and have $S(I)\ne S(I')$; and (ii) it is live at $p$: for some such pair, $f(I)(p)\ne f(I')(p)$.

**Proposition 2 (violation).** If any parameter of $f$ is a non-degenerate functional $S(I)$ of the whole image (e.g. a quantile of a dispersion map, or a $P95$ normalisation), then $f$ is not local, and there exist images and partitions for which $f_{\mathcal P_n,h}(I)\ne f(I)$ for every $h$.

*Proof.* Fix a partition $\mathcal P_n$, a halo $h$ and a tile $T$; write $T\oplus h$ for the tile extended by the halo, and let $D=\operatorname{diam}(T\oplus h)$. By condition (i) there are images $I,I'$ that agree on the ball of radius $D$ about a pixel $p\in T$ while $S(I)\ne S(I')$. Since $T\oplus h$ lies inside that ball, the two images present *the same array* to the tile-wise pass, so $f_{\mathcal P_n,h}(I)(p)=f_{\mathcal P_n,h}(I')(p)$. By condition (ii), however, $f(I)(p)\ne f(I')(p)$. Hence $f_{\mathcal P_n,h}$ differs from $f$ on at least one of the two images, and the argument holds for every $h$. $\square$

The two propositions give the taxonomy used throughout the paper.

- **Class-A mismatch (support truncation).** $S$ is absent; the difference between tiled and whole evaluation is confined to a band of width $\le B$ around tile borders and vanishes exactly at $h\ge B$.
- **Class-B mismatch (statistic non-locality).** $S$ is a whole-image functional; the difference persists in tile interiors at every $h$, and its magnitude is governed by how much $S$ varies between tiles.

Note the distinction from *non-local* filters such as non-local means: there the *support* is genuinely large, so class A is the whole story and consistency is restored by a large halo. Our claim concerns operators whose support is small but whose *parameter* is global, a case in which no halo works.

## 3.1 The algebra of the decision statistic: an exact rank bound, and what it does not buy

Everything above is about *where* a parameter looks. This subsection is about *what it is*, because the two questions are conflated in the adaptive-filter literature and the conflation is worth closing explicitly. An AKF-family operator decides, per pixel, from the windowed second moment of the image gradient:

$$M(x) = \sum_{q} w(q-x)\, g(q)\, g(q)^{\mathsf T}, \qquad g(q) = \nabla I(q) \in \mathbb{R}^{2} \qquad (1)$$

which is the structure tensor of the window. Each summand is a rank-1 symmetric PSD matrix, so $M(x)$ is symmetric PSD in a 2-dimensional space:

**Property 1 (dimension bound).** $\operatorname{rank} M(x) \le 2$ for every $x$, every weight and every image. The Eckart–Young rank-1 truncation of $M(x)$ is available in closed form from its eigenvalues $\lambda_{\max}(x) \ge \lambda_{\min}(x) \ge 0$ with eigenvector $u(x)$:

$$\left\| M(x) - \lambda_{\max}(x)\, u(x) u(x)^{\mathsf T} \right\|_{2} = \lambda_{\min}(x), \qquad \kappa(x) = \frac{\lambda_{\min}(x)}{\lambda_{\max}(x)} \in [0,1] \qquad (2)$$

$\kappa$ is exactly the anisotropy/coherence quantity that the base operator already thresholds [@kyprianidis2009akf], and $\operatorname{rank} M(x) = 1$ holds *exactly* when the gradients inside the window are mutually collinear, a single ideal straight edge, nothing else.

Property 1 is a statement about the ambient dimension, not about compressibility, and the distinction decides whether the operator's state can be shrunk. We therefore measured $\kappa$ rather than assuming anything: box weights of radius 8 (the base support), Sobel gradients, per-pixel $2\times2$ eigen-decomposition, over 10 images at $384\times384$ (these are ten `met_01000xx` records, disjoint from the ten images of Sections 6.1–6.5), discarding pixels with $\lambda_{\max} \le 10^{-12}$ (89.5–100 % of pixels are active). The numbers below are the fraction of active pixels:

**Table 2.** Measured coherence ratio $\kappa=\lambda_{\min}/\lambda_{\max}$ of the windowed gradient second moment $M(x)$ over 10 `met_01000xx` images at $384\times384$ (box weights of radius 8, Sobel gradients, per-pixel $2\times2$ eigen-decomposition, pixels with $\lambda_{\max}\le10^{-12}$ discarded): share of active pixels with $\kappa\le0.01$, $\kappa\le0.05$ and $\kappa>0.50$, the per-image median and 90th percentile of $\kappa$, and the share of exactly $\lambda_{\min}=0$ (true rank 1); every value is the range over the 10 images.

| statistic, range over the 10 images | value |
|--------------------------|--------------------------|
| $\kappa \le 0.01$ (rank-1 truncation loses ≤ 1 % of the spectrum) | 0.10 – 2.36 % (median 1.32 %) |
| $\kappa \le 0.05$ | 3.98 – 12.08 % |
| $\kappa > 0.50$ (two comparable directions) | 17.08 – 70.96 % |
| median $\kappa$ per image | 0.268 – 0.709 |
| $90$th percentile $\kappa$ per image | 0.591 – 0.888 |
| exactly $\lambda_{\min} = 0$ (true rank 1) | **0.00 % on every image** |

Read against Property 1, the claim a "low-rank state" argument needs is **false for this operator on this data**. Rank deficiency is a measure-zero event in real imagery, exactly rank-1 on 0.0 % of active pixels in every one of our ten images (Table 2), $1\,474\,560$ pixels in all, and the rank-1 truncation is *not* a small perturbation over the overwhelming majority of the canvas. What survives is weaker but true and, unlike the low-rank reading, checkable: the state of the decision statistic is bounded by the dimension of the image domain ($2\times2$, and in general $d\times d$ for $d$-dimensional signals), its rank-1 truncation error has the closed form $\lambda_{\min}$ in (2), and no factorisation of $M$ reduces either the per-pixel work or the memory the pipeline needs. We state this because it is the obvious-looking theoretical extension of this line of work, and it does not hold.

What Property 1 *does* connect to is Section 3's taxonomy. $\kappa(x)$ is a unit-free, per-pixel ratio whose value is set by image content; CA-AKF normalises a percentile of it over the whole image and SC-AKF thresholds a quantile of the dispersion map. A global functional of a content-determined per-pixel distribution is precisely the object Proposition 2 says cannot be reproduced tile-wise, and Section 6.3 measures what it becomes when each tile computes its own.

## 3.2 What tiled evaluation costs: an upper bound, and what the bound buys

Let the canvas be $S\times S$, tiled into $N = (S/T)^2$ tiles of side $T$ with overlap halo $h$, and let $W_{\mathrm{whole}}$ be the cost of one whole-canvas pass. Split the operator's cost into a per-position term and a per-tile fixed term $c_0$ (padding, allocation of the moment arrays, copying the tile's output into the canvas).

**Proposition 3 (cost bound).** If the per-position cost is uniform across the canvas,
$$W_{\mathrm{tiled}} \le \left(1 + \frac{2h}{T}\right)^{2} W_{\mathrm{whole}} + N\,c_0 \qquad (3)$$

*Proof.* Each tile emits $T^{2}$ pixels after evaluating $(T+2h)^{2}$ positions, so the tiled pass evaluates $N(T+2h)^{2} = (1+2h/T)^{2} S^{2}$ positions. At uniform per-position cost this is $(1+2h/T)^{2}$ times the $S^{2}$ positions of the whole pass, and the per-tile fixed cost is charged $N$ times. $\square$

**Proposition 4 (memory bound).** The tiled pass holds one tile's working set at a time, so $\mathrm{peak}_{\mathrm{tiled}} \le \mathrm{array} + c_{\mathrm{ws}}(T+2h)^{2}$ while $\mathrm{peak}_{\mathrm{whole}} \ge c_{\mathrm{ws}}' S^{2}$: the tiled footprint is set by $T$, not by $S$, and when canvas-sized intermediates dominate the ratio is $\approx (S/(T+2h))^{2}$ up to the retained-array term.

Two consequences are worth stating before the measurements, because they are the opposite of what a "tiling accelerates large images" framing would predict.

*First, (3) contains no regime in which tiling is faster.* The multiplicative term is $\ge 1$ and the additive term is $\ge 0$; the bound is an upper bound on a cost that is always larger than the whole-canvas cost. The content of (3) is therefore budgetary, not asymptotic: it says *exactly how much slower* a memory-feasible pipeline is, as a known function of the halo and the tile count, and it says which knob to turn when the budget is tight. Halving $T$ at fixed $h$ raises the time multiplier and lowers the memory peak; that trade is the whole of what tiling offers.

*Second, combine (3) with Propositions 1–2 and the paper's asymmetry becomes arithmetic.* For a class-A operator, the halo $h$ in (3) is the price of exactness and the price stops at $h = B$: beyond the support the multiplier buys nothing further, and Section 6.7 measures exactly this (AKF's deviation is literally zero at $h=48 > B$, and 0.12 of a display level at $h=16$). For a class-B operator, $h$ does not converge at all, so the multiplier in (3) is paid with no correctness return, and the deviation depends on the tile *partition*, which is why a pipeline that switches from $n=2$ to $n=4$ to save memory silently renders a different image (Section 6.7).

Measured against (3) on the $2048\times2048$ mosaic of Section 6.7 (one mode per process, halo 48; six of the seven pairs were timed inside a single session, so the ratios are not artefacts of host load, while the seventh (the LRS-AKF $n=4$ row, whose whole pass is a manual run) is cross-session and carries no predictive content):

**Table 3.** Measured whole-image and tile-wise wall time (s) for AKF, CA-AKF, SC-AKF and LRS-AKF (ours) on the $2048\times2048$ mosaic of Section 6.7 (one mode per process, halo 48) against the Proposition 3 bound $(1+2h/T)^2$, on tile grids $n=4$ ($N=16$ tiles) and $n=8$ ($N=64$ tiles): measured/bound ratio, whether the multiplicative term alone is satisfied, and the per-tile fixed cost $c_0$ required by the bound and fitted (s/tile).

| method | grid $n$ ($N$ tiles) | whole s | tile-wise s | measured / bound | multiplier alone holds? | $c_0$ required (s/tile) | fitted $c_0$ (s/tile) |
|--------------|----------------|-------|-----------|----------------|-----------------------|--------------------|------------------|
| AKF | 4 (16) | 33.9 | 46.0 | 1.359 / 1.410 | **yes** | 0.000 | 0.150 |
| AKF | 8 (64) | 33.9 | 73.6 | 2.175 / 1.891 | no | 0.150 | 0.150 |
| CA-AKF | 4 (16) | 143.9 | 205.9 | 1.431 / 1.410 | no | 0.190 | 0.761 |
| CA-AKF | 8 (64) | 143.9 | 320.7 | 2.229 / 1.891 | no | 0.761 | 0.761 |
| SC-AKF | 4 (16) | 356.6 | 516.5 | 1.448 / 1.410 | no | 0.852 | 1.132 |
| SC-AKF | 8 (64) | 356.6 | 746.6 | 2.094 / 1.891 | no | 1.132 | 1.132 |
| LRS-AKF (ours) | 4 (16) | 232.8 | 344.2 | 1.479 / 1.410 | no | 1.000 | 1.000 |

The multiplicative term alone is satisfied by one of seven pairs (Table 3); every class-B operator and every finer grid needs the additive term, and one fitted $c_0$ per method then covers all of its pairs. Three things in that column matter. (i) For AKF, CA-AKF and SC-AKF the fit is taken from the *finer* grid and the *coarser* grid's requirement is lower (0.000, 0.190, 0.852 s ≤ 0.150, 0.761, 1.132 s), so the bound predicts at one spacing rather than interpolating. LRS-AKF has one grid measured (its $n=8$ run was lost to a restart), so there fit and check coincide and no predictive content is claimed. (ii) The size of $c_0$ is mechanism-consistent: 0.15 s/tile for the fixed-radius filter is allocation and copying, while 0.76 s (CA-AKF) and 1.13 s (SC-AKF) carry a per-tile pass over the tile's own statistic, class B in a timing column (Section 6.3). (iii) Measured peak-RSS ratios are 4.9–8.9× (Proposition 4's shape: 2.2–3.9 GB whole against 0.36–0.63 GB tile-wise), and at $4096\times4096$ the CA-AKF whole pass is not slow but unavailable (Section 6.7, the only method tried there). The timing column needs one caveat that the bound itself does not carry.

Three separate CA-AKF whole-canvas passes at $2048^2$ exist, at 143.88 s, 92.51 s and 90.62 s; the pair quoted in Table 3 is the first, and the three sessions are priced differently because they ran under different host load. Six of the seven pairs in Table 3 were timed inside a single session and the seventh (LRS-AKF, $n=4$) was not. Each released record carries that flag per pair, so a reader can recompute every ratio from one session if they prefer. The theory section's one-line summary is therefore: *tiling buys memory and never time, the price is (3), and for class-B operators the price buys no correctness either.*

# 4 Measuring what current metrics cannot see

We need measurements that (i) detect class-B mismatch, (ii) attribute it correctly, and (iii) demonstrate its consequences.

## 4.1 Tile-consistency error (TCE)

For an output difference $D=\lvert f(I)-f_{\mathcal P_n,h}(I)\rvert$ taken as the maximum over channels, we report

- $\mathrm{RMSE}(D)$ over the image,
- $\max D$ (worst single pixel),
- $\mathrm{frac}_{1/255}(D)$: the fraction of pixels differing by more than one 8-bit display level,
- $\mathrm{frac}_{4/255}(D)$ for a coarser threshold.

Throughout, **exact** means the float-domain criterion $\max D<1/255$ with $\mathrm{frac}_{1/255}=0$; outputs are float32 and are never quantised, so no byte-level equality is claimed.

## 4.2 Border-band decomposition

Given the tile grid, let $d(p)$ be the distance from pixel $p$ to the nearest grid line. We report the mean of $D$ in the bands $d\in[0,2),[2,6),[6,12),[12,24),[24,\infty)$. **Diagnostic rule:** if the deviation decays to zero as the band index grows, the mismatch is class A; if it is flat across bands, the mismatch is class B and lives in tile interiors.

## 4.3 Grid-line visibility (GV)

$G$ is the fraction of deviation energy $\sum D^2$ that sits exactly on grid rows/columns, and the *concentration ratio* is $G$ divided by the fraction of pixels that are on the grid. A ratio $\gg 1$ means the artefact is a seam; a ratio $\approx 1$ means it is content-driven and cannot be hidden by seam post-processing.

## 4.4 Rank-inversion test

For each image and each quality metric, we rank the methods twice: once using whole-image evaluation, once using tile-wise evaluation (fixed $n$, fixed halo). We report the number of inversions and the mean Kendall $\tau$ between the two rankings. A protocol whose outcome depends on this choice is not a valid comparison of methods. This test is aimed at a specific current practice: the abstraction/style-transfer evaluation literature is actively converging on standardised metric batteries and benchmark sets [@zhou2025wise], and those batteries are computed on whole images, which is exactly the assumption our test probes.

## 4.5 Mechanism probe

For any operator whose parameter is $\tau=c\,Q_p(v)$ with $Q_p$ a whole-image quantile of a dispersion map $v$, we compute the ratio of $\tau$ on a tile to $\tau$ on the whole image, the direct, parameter-level statement of class B. The implementation is `tau_drift` in `src/tc.py`; its per-tile ratios are not in the released record set, so Section 6.3 quotes no numbers from it.

# 5 LRS-AKF: a tile-consistent content-adaptive criterion

## 5.1 The global-quantile criterion

Let $v_r(x)$ be the minimum sector variance at radius $r$ around pixel $x$ after orientation-preserving smoothing, and let $v_0$ correspond to the smallest radius $r_0$ in a geometric ladder $r_0<r_1<\dots<r_{L-1}$. The scale criterion we build on selects, for each pixel, the largest level whose dispersion still fits under a threshold:

$$\tau = c\,Q_p(v_0), \qquad \ell(x)=\max\{i: v_{r_i}(x)\le\tau\}, \qquad (4)$$

with $Q_p$ the $p$-quantile **over the whole image**, and the output taken as the soft-min pooling of the sector means at level $\ell(x)$. Published content-adaptive variants additionally normalise the structure tensor's major eigenvalue by its 95th percentile, which is a second whole-image functional.

## 5.2 Replacing the global functional

LRS-AKF keeps the semantics of the criterion (*how far may the dispersion grow relative to the baseline texture scale?*) and changes only where the reference comes from:

$$\tau(x) = c\cdot R(x), \qquad R = \mathcal B_{\rho}\!\big[\tilde v_0\big], \qquad (5)$$

where $\tilde v_0 = v_0$ (arithmetic reference) or $\tilde v_0=\log(v_0+\epsilon)$ with $R=\exp(\cdot)$ (geometric reference), and $\mathcal B_\rho$ is a box average of radius $\rho$. Three properties follow immediately:

1. **Locality / exact tile consistency.** $R(x)$ depends on $v_0$ within $\rho$ of $x$, and $v_0$ is itself local, so the whole chain has support $B^{*}=B+\rho$. $B$ is the base pipeline's *nominal* support: we use $B=16$ px, behind Section 6.6's predicted column, while Section 6.8.3 quotes the alternative reading $2r+\rho_{\text{base}}=11$ px; both are nominal, since the implementation contains unbounded Gaussian presmoothing and the true footprint must be *measured*. By Proposition 1, tiling with $h\ge B^{*}$ reproduces the whole-image result exactly; $\rho$ is the *only* new constant, and it is a memory budget, not a quality knob.
2. **Dimensionlessness.** The criterion compares a dispersion to a dispersion. No unit is inherited from the image as a whole, so the decision is unchanged by adding or removing content elsewhere in the image.
3. **Streaming compatibility.** $\mathcal B_\rho$ is a separable, bounded-support scan; it can be evaluated in one pass over tiles in raster order with $\rho$ rows of state, unlike $Q_p$, which needs the whole image (and, if approximated by a streaming summary [@greenwald2001gk], remains non-local by construction).

## 5.3 Implementation

We build on an existing anisotropic Kuwahara pipeline (structure tensor; orientation-preserving anisotropic smoothing; $K=8$ oriented sectors over $M=12$ orientation bins; soft-min pooling with temperature $\beta$; radii $\{2,4,6,8,10\}$) with all pixel-space settings identical to the baselines, so that the *only* difference between the compared operators is the origin of $\tau$. LRS-AKF adds one box average of the base dispersion map: $O(HW)$ time, $O(\rho)$ extra rows of state, no additional convolutions.

# 6 Results

All results in this section are measured on public-domain images from The Metropolitan Museum of Art Open Access programme (CC0; licence evidence recorded per image), resized to $384\times384$ for the main sweep. Comparisons are paired across the same 10 images. We use two-sided Wilcoxon signed-rank tests and adjust within two **pre-specified families** with the Holm step-down procedure: the *consistency family* (RMSE, $\mathrm{frac}_{1/255}$, $\mathrm{frac}_{4/255}$, mean deviation, worst-pixel deviation) and the *quality family* (SR, SSIM, FSR, ECE, SCG, OVT). With $n=10$ the smallest attainable two-sided $p$ is $0.00195$; we therefore treat the number of images as the binding limitation on statistical resolution, not the choice of test, and we report the per-image win rate, the paired effect size and a paired-bootstrap interval alongside every $p$-value in the quality family. Uncorrected and Holm-adjusted values are both given in the Section 6.4 tables so that no claim depends on the multiplicity convention.

## 6.1 The global-quantile configuration is not tile-consistent, and haloes do not help

Table 4 gives TCE (median RMSE) as a function of the overlap halo for a $4\times4$ grid, and Figure 1 plots the same ladder: the two local operators fall to machine exactness at their support width, the two global-quantile operators plateau.

**Table 4.** Tile-consistency error on the 10 images at $384\times384$: median RMSE of the tiled-versus-whole output difference $D$ on a $[0,1]$ scale as a function of the overlap halo (0, 8, 16, 24 and 32 px) on a $4\times4$ tile grid, with the fraction of pixels differing by more than one 8-bit level and the worst single-pixel deviation $\max D$ at halo 32, for AKF (fixed radius, local control), CA-AKF (global-quantile reference), SC-AKF (global quantile, tuned), LRS-AKF (ours, arithmetic reference) and LRS-AKF (ours, final configuration); medians over the 10 images.

| method | halo 0 | halo 8 | halo 16 | halo 24 | halo 32 | frac>1/255 @ halo 32 | worst $\max D$ @ halo 32 |
|--------------------------|-------|-------|-------|--------|--------|--------------------|--------------------|
| AKF (fixed radius, local control) | 8.28e-3 | 1.10e-3 | 2.31e-5 | 8.73e-11 | **0.0** | 0.0000 | 9e-13 |
| CA-AKF (global-quantile reference) | 5.69e-3 | 2.48e-3 | 2.25e-3 | 2.11e-3 | **1.99e-3** | **0.0652** | 8.28e-2 |
| SC-AKF (global quantile, tuned) | 1.87e-2 | 1.70e-2 | 1.47e-2 | 1.39e-2 | **1.34e-2** | **0.2055** | 6.18e-1 |
| LRS-AKF (ours, arithmetic ref.) | 9.55e-3 | 1.06e-3 | 1.50e-4 | 1.12e-7 | **4.31e-12** | 0.0000 | 8.2e-4 |
| LRS-AKF (ours, final config) | 5.49e-3 | 6.91e-4 | 5.74e-5 | 5.98e-8 | **2.12e-9** | 0.0000 | 4e-6 |

Median over the 10 images; $\max D$ is the worst single pixel in the worst image.

The two local operators (the fixed-radius control and ours) decay to *machine exactness*, as Proposition 1 predicts. At halo 32 the tiled LRS-AKF output differs from the whole-image output by at most $4\times10^{-6}$ on a $[0,1]$ scale (one thousandth of one 8-bit level) on every one of the 10 images. The fixed-radius control reaches zero on 8 of 10 and $9\times10^{-13}$ on the other two. (At the scale of Section 6.8 this is not universal: 3 of 144 images leave a *single* pixel at 1.2–3.3 levels after a 32-pixel halo, named in Section 6.8.4.) The two global-statistic operators plateau: doubling the halo from 16 to 32 px reduces their TCE by only 9–14 % (median RMSE: CA-AKF $-11.7$ %, SC-AKF $-9.0$ %; $\mathrm{frac}_{1/255}$: $-13.9$ % and $-12.2$ %), and at halo 32 **6.5 % of pixels (CA-AKF) and 20.5 % (SC-AKF) still differ by more than one 8-bit level**. Their worst single-pixel deviations are 0.083 and 0.618 on a $[0,1]$ scale, 21 and 158 display levels, i.e. a visibly different brush size at isolated pixels.

## 6.2 The mismatch is in the tile interiors, not on the seams

Border-band decomposition at $n=8$, halo 16, plotted in Figure 2 (mean $D$ per band, median over images):

**Table 5.** Border-band decomposition at $n=8$ and halo 16 px on the 10 images at $384\times384$: mean deviation $D$ (on a $[0,1]$ scale) in bands of distance from the nearest tile grid line (0–2, 2–6, 6–12, 12–24 and $>24$ px), median over images, for AKF, CA-AKF, SC-AKF and LRS-AKF (final); the same data are plotted in Figure 2.

| method | 0–2 px | 2–6 | 6–12 | 12–24 | >24 |
|---------------|-------|-------|-------|-------|-------|
| AKF | 5.2e-6 | 0 | 0 | 0 | 0 |
| CA-AKF | 1.75e-3 | 1.80e-3 | 1.79e-3 | 2.00e-3 | **1.75e-3** |
| SC-AKF | 8.92e-3 | 9.68e-3 | 8.13e-3 | 8.10e-3 | **9.11e-3** |
| LRS-AKF (final) | 1e-5 | 0 | 0 | 0 | 0 |

CA-AKF's deviation is *flat* with distance from the grid (Table 5): it does not decay: that is the signature of class B. Consistently, grid-line visibility at $n=4$, halo 16 is 0.016 (concentration ratio 1.0×) for CA-AKF and 0.017 (1.1×) for SC-AKF, versus 0.221 (14.2×) for the fixed-radius control, i.e. the control's residual error is a genuine seam that a seam-blender could hide, whereas the global-statistic error is spread through the image and cannot. For our operator (final configuration) GV is 0.090 (5.8×) at halo 16 (the small remaining error is seam-localised (class A)) and by halo 24 it has shrunk to a single pixel of one image (`met_0435751.jpg`, 1.83 levels above one display level), with none left at halo 32. The arithmetic configuration of the same operator gives 0.062 (4.0×), a row that our analysis table prints without labelling the configuration.

## 6.3 Mechanism: the threshold is a different number in every tile

Class B is about the parameter, not the pixels: $\tau=c\,Q_p(v_0)$ takes one value for the whole image and another for every tile, so no halo can make a tile's $\tau$ equal the whole-image $\tau$: the halo supplies pixels, and the statistic was never a function of nearby pixels. Sections 6.1 and 6.2 measure the consequence (a deviation that neither decays with the halo nor concentrates at the grid), and Section 3.2's per-tile fixed costs (0.76 s, 1.13 s against 0.15 s) name its mechanism: each tile's extra pass over its own statistic. We quote no per-tile ratios of $\tau$: the probe that would produce them has no caller in the release and no released record, so Section 4.5's statement is left as a test a reader can run.

## 6.4 Quality: consistency costs nothing and buys improvement

Paired comparison over the same 10 images (mean ± sd; direction: SR/SSIM/SCG higher is better, FSR/ECE/OVT lower is better; $p$ from two-sided Wilcoxon signed-rank, Holm-adjusted within the six-metric quality family):

**Table 6.** Paired quality comparison of CA-AKF (global-quantile reference) against LRS-AKF (final) over the same 10 images at $384\times384$: mean $\pm$ sample sd, the paired mean difference $\Delta$, the paired effect size $d_z$ (mean difference divided by the sample sd of the paired differences), a 95 % paired-bootstrap interval for $\Delta$ (20,000 resamples, fixed seed 20261104), the number of images won, and the two-sided Wilcoxon signed-rank $p$ with its Holm-adjusted value. SR/SSIM/SCG higher is better; FSR/ECE/OVT lower is better.

| metric | CA-AKF | LRS-AKF | $\Delta$ | $d_z$ | 95 % CI of $\Delta$ | win/10 | $p$ | $p_{\mathrm{Holm}}$ |
|------|--------------|--------------|--------------|-----|------------------|------|-------|--------|
| SR | 0.5353 ± 0.087 | **0.5656 ± 0.097** | +0.030 | +0.68 | [+0.0020, +0.0544] | 8/10 | 0.08398 | 0.168 |
| SSIM | 0.6904 ± 0.079 | **0.7259 ± 0.081** | +0.036 | +3.41 | [+0.0295, +0.0416] | 10/10 | 0.00195 | **0.012** |
| FSR | **0.1981 ± 0.056** | 0.2287 ± 0.065 | +0.031 (worse) | +2.64 | [+0.0236, +0.0371] | 0/10 | 0.00195 | **0.012** |
| ECE | 18.08 ± 4.178 | **17.20 ± 4.104** | -0.879 | -1.05 | [-1.3795, -0.4038] | 7/10 | 0.02734 | 0.082 |
| SCG | 0.0756 ± 0.017 | **0.0858 ± 0.017** | +0.010 | +1.01 | [+0.0047, +0.0165] | 8/10 | 0.00977 | **0.039** |
| OVT | 0.0069 ± 0.008 | **0.0043 ± 0.004** | -0.003 | -0.52 | [-0.0057, -0.0001] | 6/10 | 0.13672 | 0.168 |

With family-wise adjustment at $\alpha=0.05$, LRS-AKF is **significantly better on SSIM and SCG, significantly worse on FSR, and not statistically distinguishable on SR, ECE or OVT** at $n=10$ (Table 6). The claim we need is not "better on every metric": it is that exact tile consistency costs **no measurable quality** on five of the six metrics and one measurable regression, FSR, analysed in the next paragraph. The two surviving gains are large (SSIM +5.1 % relative, SCG +13.5 % relative) for an operator that changes one line of the parameter chain.

At $n=10$ the Wilcoxon statistic sits on its resolution floor, so a $p$-value alone cannot separate a large effect from a small one. Table 6 therefore reports the paired effect size $d_z$ and a paired-bootstrap interval for the mean difference alongside every $p$-value. The two gains and the one regression that survive Holm correction have $|d_z|>1$ and intervals well away from zero (SSIM $d_z=3.41$, FSR $d_z=2.64$). OVT is the informative disagreement: its mean-difference interval excludes zero ($[-0.0057,-0.0001]$) while the rank test does not reject ($p=0.137$), which is what a skewed metric with one large pair looks like. We keep OVT in the indistinguishable group because the pre-specified family test decides significance; Section 6.11 shows that this borderline interval does not survive replication. The interval describes the mean; the rank test decides the claim, and reporting both is what keeps the floor visible.

The single regression, FSR (flat-region residual variance, lower is better), is expected and is a genuine trade-off rather than a noise artefact: the local reference is a *relative* quantity, and with $c=1$ the operator is more conservative about growing the support in flat regions than the global $p25$-based threshold, so it retains more high-frequency texture where the global threshold paints regions flat. It is controllable by one knob: increasing $c$ moves FSR back toward the baseline while leaving tile consistency untouched (Section 6.6), because consistency is a property of the *support* of the parameter chain, not of its threshold value. Against the fixed-radius baseline the improvement is uniform (AKF: SR 0.400, SSIM 0.649, ECE 20.9).

For the primary claim (tile consistency) the paired difference CA-AKF → LRS-AKF at $n=8$ (the $8\times8$ tile grid; the median is over the same 10 images), halo 16, is: median RMSE $3.51\times10^{-3}\to1.01\times10^{-4}$ (34.9×), median $\mathrm{frac}_{1/255}$ $12.59\%\to0.009\%$ (1429×), median $\mathrm{frac}_{4/255}$ $1.63\%\to0.0014\%$, mean deviation $1.31\times10^{-3}\to2\times10^{-6}$, worst pixel $0.074\to0.030$. All five endpoints are won by 10/10 images with raw $p=0.00195$ and Holm-adjusted $p=0.0098$ within the consistency family; the adjusted value is the floor attainable at $n=10$, so the effect is limited by the test's resolution, not by its size.

*Are our quality numbers themselves protocol-dependent?* A quality claim that only holds under whole-image evaluation would be worth little, because the pipeline that has to run is the tiled one. Measuring every method under both protocols on the same 10 images (mean over images; $\Delta$ = tiled minus whole, so a $\Delta$ of zero at the four decimals printed means the published number is reproducible on a tiled harness):

**Table 7.** The six quality metrics (SR, SSIM, SCG, ECE, OVT, FSR) for AKF, CA-AKF, SC-AKF and LRS-AKF (ours, final configuration) measured under both whole-image and tile-wise evaluation on the same 10 images, printed as whole → tiled with $\Delta$ = tiled minus whole at four decimals.

| method | SR whole → tiled | SSIM whole → tiled | SCG whole → tiled | ECE whole → tiled | OVT whole → tiled | FSR whole → tiled |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|----------------------|
| AKF | 0.4002 → 0.4002 ($+0.0000$) | 0.6494 → 0.6494 ($-0.0000$) | 0.0221 → 0.0221 ($+0.0000$) | 20.9179 → 20.9179 ($+0.0000$) | 0.0298 → 0.0298 ($+0.0000$) | 0.2650 → 0.2649 ($-0.0001$) |
| CA-AKF | 0.5353 → 0.5357 ($+0.0004$) | 0.6904 → 0.6929 ($+0.0025$) | 0.0756 → 0.0767 ($+0.0012$) | 18.0836 → 18.0819 ($-0.0017$) | 0.0069 → 0.0059 ($-0.0009$) | 0.1981 → 0.1981 ($+0.0000$) |
| SC-AKF | 0.5147 → 0.4837 ($\mathbf{-0.0310}$) | 0.6983 → 0.6792 ($\mathbf{-0.0191}$) | 0.0686 → 0.0600 ($\mathbf{-0.0086}$) | 18.6882 → 19.0911 ($\mathbf{+0.4029}$) | 0.0321 → 0.0454 ($\mathbf{+0.0132}$) | 0.2016 → 0.1925 ($-0.0091$) |
| LRS-AKF (ours, final config) | 0.5656 → 0.5656 ($+0.0000$) | 0.7259 → 0.7259 ($+0.0000$) | 0.0858 → 0.0858 ($-0.0000$) | 17.2045 → 17.2047 ($+0.0001$) | 0.0043 → 0.0043 ($+0.0000$) | 0.2287 → 0.2287 ($+0.0000$) |

(SR/SSIM/SCG higher is better; FSR/ECE/OVT lower is better. Every non-zero $\Delta$ for SC-AKF is a degradation except FSR; every non-zero $\Delta$ for CA-AKF is an improvement of at most $0.0025$.)

Three readings (Table 7). (i) **SC-AKF's apparent quality is a property of the harness, not of the operator**: tile-wise it loses 0.031 SR, 0.019 SSIM and 0.009 SCG and worsens ECE by 0.40 and OVT by 0.013, improving only FSR. Against CA-AKF its whole-image SSIM edge is $+0.0079$; the same two operators evaluated tile-wise differ by $-0.0137$. The sign of a headline quality claim reverses under a harness change, which is Section 6.5's rank-inversion mechanism acting on the metric a reader would trust most. (ii) **CA-AKF's numbers barely move** ($\le0.0025$ on every metric) because its global functional is a percentile of a near-uniform quantity: same class, much weaker symptom. (iii) **AKF and LRS-AKF move by zero to the four decimals we report** on all six metrics, largest residual $1.2\times10^{-4}$ (LRS-AKF, ECE) and $8\times10^{-5}$ (AKF, FSR), both invisible at the printed precision, the consistency property doing something useful for reporting, not only for rendering: an exactly tile-consistent operator's published numbers cannot be perturbed by another group's evaluation harness. This is the sense in which the contribution is metrological, and the one claim here that the six quality metrics are blind to.

Note also what these columns do *not* say: they are not a quality ranking, and SC-AKF remains a legitimate perceptual configuration whose global quantile was tuned on whole-image statistics: the point is that its reported quality is not portable, while ours is.

## 6.5 The consequence for published comparisons: rank inversions

Figure 3 shows that the established quality metrics are blind to the deviation that TCE measures, and Figure 4 plots the consequence: across 4 metrics × 10 images, evaluating the *same* four filters tile-wise instead of whole-image inverts the method ranking in **15 of 40 cases**; the mean Kendall $\tau$ between the two rankings is 0.80 (FSR), 0.83 (SSIM), 0.90 (SR) and 0.93 (ECE). The four filters ranked are CA-AKF, SC-AKF, AKF and the *arithmetic* LRS-AKF configuration of Table 4, not the final one, so on more than a third of the image–metric combinations a reader reproducing a comparison on a tiled pipeline would report a different winner. The moves are of the same order as the inversions they cause: at most $0.0025$ (CA-AKF) and $1.2\times10^{-4}$ (the two bounded-support operators), while SC-AKF moves by $0.031$ SR, $0.019$ SSIM and $0.40$ ECE (Section 6.4), a four-method ranking inverts on a move no reviewer would call an effect size, which is why this has gone unnoticed.

## 6.6 The knob is a memory budget, not a quality knob

Sweeping the reference radius $\rho$ measures the bias–memory trade-off that Proposition 1 makes explicit. For each $\rho$ we compute the smallest halo (tested in steps of 8 px up to 40 px) at which the tiled output is exactly the whole-image output for every pixel (4 images at $384\times384$, $n=4$ grid):

**Table 8.** LRS-AKF reference radius $\rho$ (px) and its extra state per tile row on 4 images at $384\times384$ with a $4\times4$ tile grid: the predicted exact halo $B+\rho$ against the measured smallest halo (8-px steps up to 40 px) at which the tiled output is exactly the whole-image output for every pixel, with median TCE at halo 32 and SR, SSIM and ECE.

| $\rho$ (px) | extra state per tile row | predicted exact halo $B+\rho$ | measured exact halo | median TCE @ halo 32 | SR | SSIM | ECE |
|--------|------------------------|-------------------------|--------------------------|--------------------|-----|-----|-----|
| 4 | 4 rows | 20 | **24 (4/4 images)** | 0.0 (max $1\times10^{-8}$) | 0.543 | 0.744 | 16.25 |
| 8 | 8 rows | 24 | **24 (4/4)** | 0.0 (max $1\times10^{-8}$) | 0.540 | 0.741 | 16.24 |
| 16 | 16 rows | 32 | **32 (4/4)** | 0.0 (max $3.9\times10^{-6}$) | 0.536 | 0.739 | 16.22 |
| 32 | 32 rows | 48 | 40 (2/4), 24 (1/4), none at $\le 40$ px (1/4) | $2.0\times10^{-4}$ | 0.537 | 0.740 | 16.17 |

Two findings matter here, and not equally (Table 8). First, **the measured exact halo follows $B+\rho$ wherever the 8-px ladder resolves it**: predicted 20, 24, 32, 48 px against measured 24, 24, 32 px for $\rho=4,8,16$ (4/4 images at the last two; at $\rho=4$ the prediction falls between rungs, so the measured value is the next rung above). At $\rho=32$ the prediction is past the 40-px ceiling and the images split: two exact at 40 px, one at 24 px (earlier than $B+\rho$, which a sufficient condition permits), one at no tested halo. Resolved increments of 0, $+8$, $+8$ px against $\Delta\rho$ of $+4$, $+8$, $+16$ px make $B+\rho$ a first estimate of the halo, not an upper bound, but still a *memory budget with a known price*: pick an affordable halo and verify exactness at it per image (Section 6.8.3) rather than tuning by eye until seams disappear. Second, **quality is nearly flat in $\rho$** (SR 0.543 → 0.537, SSIM 0.744 → 0.740, within 1.3 % relative over an eight-fold change in radius): the reference window is a *correctness* parameter, not a *look* parameter, unlike the global quantile it replaces, which tracks image content and so silently changes the look of every tile.

The same independence holds for the threshold multiplier $c$: changing $c$ rescales $\tau(x)$ and therefore moves the smoothness/texture trade-off, but it does not change the support of the parameter chain and therefore cannot change the halo at which exactness is attained. We verified this on 4 images $\times$ $c \in \{1, 1.5, 2, 3\}$ at $\rho = 8$ (16 configurations, $n=4$):

**Table 9.** Threshold multiplier $c\in\{1,1.5,2,3\}$ for LRS-AKF at $\rho=8$ on 4 images with a $4\times4$ tile grid (16 configurations): median SR, SSIM and ECE, the number of configurations with $\max D<1$ 8-bit level at halo 24 px, and the worst-pixel deviation (levels) and worst RMSE at halo 32 px.

| quantity | $c=1$ | $c=1.5$ | $c=2$ | $c=3$ |
|--------------------------|--------|--------|--------|--------|
| median SR (look) | 0.540 | 0.453 | 0.419 | 0.393 |
| median SSIM (look) | 0.741 | 0.696 | 0.673 | 0.657 |
| median ECE (look) | 16.24 | 17.74 | 18.13 | 18.40 |
| configurations with $\max D < 1$ level at halo 24 | 4/4 | 4/4 | 4/4 | 3/4 |
| worst $\max D$ at halo 32 (levels) | $5.6\times10^{-4}$ | $5.6\times10^{-4}$ | $5.6\times10^{-4}$ | $5.6\times10^{-4}$ |
| worst RMSE at halo 32 | $6.0\times10^{-9}$ | $6.0\times10^{-9}$ | $6.0\times10^{-9}$ | $6.0\times10^{-9}$ |

So $c$ is a *look* knob (SR falls 27 % relative from $c=1$ to $c=3$) that is not a *correctness* knob (Table 9). At the predicted halo $\rho+B = 24$ px the tiled result is already within one 8-bit level of the whole-image result in 15 of 16 configurations, and within $8.4\times10^{-3}$ (2.2 levels) in the sixteenth. By 32 px every configuration has collapsed to $6\times10^{-9}$ RMSE and $2.2\times10^{-6}$ worst pixel, machine noise, and *the same worst-pixel value for every $c$*, because beyond the support bound the two computations evaluate the same expression on the same inputs.

That last column also warns against a criterion we deliberately did **not** use as the headline. The halo at which the deviation first drops *below one 8-bit level* is not stable: measured that way it is 24 px for 12 configurations, 16 px for 3 and 32 px for 1, because before the support bound is reached the residual is a rounding-scale artefact whose magnitude depends on which pixels change their selected support. A "visually indistinguishable overlap" is therefore a property of the quantisation grid and the image, not of the operator, which is exactly why we report the analytically predicted halo plus machine-scale exactness rather than a per-image threshold crossing.

## 6.7 Large canvases

To test the deployment claim we assemble a $2048\times2048$ mosaic of the public-domain images (50.3 MB as a float32 RGB array) and compare a whole-canvas pass with tile-wise evaluation *on the same pixels*, one method per process so that the reported memory belongs to that method. The cost table below (Table 10) come from two timing sessions and the deviation table (Table 11) from a third; the CA-AKF whole passes in them are priced differently (Section 3.2 lists all three).

Whole-canvas versus tile-wise cost, measured with **one mode per process** so that each peak belongs to the pass it names ($512\times512$ tiles, halo 48):

**Table 10.** Filter time (s) and peak RSS (MB) for whole-canvas and tile-wise ($n=4$, $n=8$) passes of AKF, CA-AKF, LRS-AKF (ours) and SC-AKF on the assembled $2048^2$ mosaic at $512\times512$ tiles and halo 48, one mode per process, plus the CA-AKF pair at $4096^2$ where the whole-canvas pass was terminated by the kernel OOM killer on an 8 GB host.

| canvas | method | mode | filter time | peak RSS |
|------|--------------|------------|--------------------------|-----------------|
| $2048^2$ | AKF | whole | 33.9 s | 2174 MB |
| $2048^2$ | AKF | tile-wise $n=4$ | 46.0 s | **422 MB** |
| $2048^2$ | AKF | tile-wise $n=8$ | 73.6 s | 377 MB |
| $2048^2$ | CA-AKF | whole | 143.9 s | 2176 MB |
| $2048^2$ | CA-AKF | tile-wise $n=4$ | 205.9 s | **447 MB** |
| $2048^2$ | CA-AKF | tile-wise $n=8$ | 320.7 s | 361 MB |
| $2048^2$ | LRS-AKF (ours) | whole | 232.8 s | 3934 MB |
| $2048^2$ | LRS-AKF (ours) | tile-wise $n=4$ | 344.2 s | **629 MB** |
| $2048^2$ | SC-AKF | whole | 356.6 s | 3847 MB |
| $4096^2$ | CA-AKF | whole | *not completed*, terminated by the kernel OOM killer (exit 137) on an 8 GB host | > 8 GB (exceeded) |
| $4096^2$ | CA-AKF | tile-wise $n=8$ | 833.0 s | **995 MB** |

A 50 MB image costs 2.2–3.9 GB of resident memory to filter in one piece (43× to 78× the array) because the operator materialises several float32 intermediates at canvas size; the same computation tile-wise costs 0.36–0.63 GB on this canvas (the largest tile-wise figure in the table, 0.97 GB, is the $4096^2$ pass), so its footprint is set by the tile rather than the canvas, at a 1.36–1.48× time overhead. Times are load-dependent (shared four-core host); peak RSS, reported per process, is the quantity the argument needs. The last two rows are the point of the section: one size up ($4096^2$, 201 MB array) the CA-AKF whole-canvas pass *cannot be run at all* on this machine (the only method we ran at that size) while the tile-wise pass completes in 995 MB. The asymmetry is worth stating plainly: because there is no whole-canvas reference at $4096^2$, the *deviation* is not measured there but at $2048^2$, where both passes exist, so the $4096^2$ run establishes feasibility only. The evaluation the literature treats as ground truth is unavailable exactly where the claim has to hold, and the pipeline that *can* run is the one that, for a class-B operator, returns a different image.

Tile-wise output versus the whole-canvas output on that mosaic:

**Table 11.** Tile-wise output versus the whole-canvas output on the $2048\times2048$ mosaic for AKF, CA-AKF, SC-AKF and LRS-AKF (ours) at tile grids $n=2$ and $n=4$ with halos 16 and 48 px: RMSE, worst-pixel deviation in 8-bit levels, fraction of pixels differing by more than one and by more than four levels, and the grid-line visibility ratio.

| method | grid / halo | RMSE | worst pixel (levels) | frac $>1$ level | frac $>4$ levels | GV ratio |
|--------------|-----------|-------|--------------------|--------------------------|--------------|--------|
| AKF | $n=2$, halo 16 | 0.0 | 0.12 | 0.00 % | 0.00 % | 33.1 |
| AKF | $n=2$, halo 48 | 0.0 | 0.00 | 0.00 % | 0.00 % | 0.0 |
| AKF | $n=4$, halo 16 | 0.0 | 0.12 | 0.00 % | 0.00 % | 255.8 |
| AKF | $n=4$, halo 48 | 0.0 | 0.00 | 0.00 % | 0.00 % | 0.0 |
| CA-AKF | $n=2$, halo 16 | 1.81e-3 | 13.0 | 4.87 % | 0.30 % | 2.0 |
| CA-AKF | $n=2$, halo 48 | 1.73e-3 | 13.5 | 4.57 % | 0.27 % | 2.3 |
| CA-AKF | $n=4$, halo 16 | 3.22e-3 | 27.7 | 5.65 % | 0.88 % | 0.7 |
| CA-AKF | $n=4$, halo 48 | 2.86e-3 | 25.3 | 5.08 % | 0.70 % | 0.7 |
| SC-AKF | $n=2$, halo 16 | 1.09e-2 | 123.1 | 15.26 % | 6.23 % | 3.8 |
| SC-AKF | $n=2$, halo 48 | 1.05e-2 | 123.1 | 14.22 % | 5.88 % | 4.0 |
| SC-AKF | $n=4$, halo 16 | 1.85e-2 | 175.2 | 29.49 % | 13.73 % | 5.6 |
| SC-AKF | $n=4$, halo 48 | 1.49e-2 | 145.3 | 24.10 % | 10.33 % | 5.1 |
| LRS-AKF (ours) | $n=2$, halo 16 | 8e-6 | 3.3 | $<5\times10^{-7}$ (at most 2 px of 4.2 M) | 0.00 % | 0.1 |
| LRS-AKF (ours) | $n=2$, halo 48 | **0.0** | 0.004 | 0.00 % | 0.00 % | 0.0 |
| LRS-AKF (ours) | $n=4$, halo 16 | 1.1e-5 | 4.9 | $1\times10^{-6}$ (4 px of 4.2 M) | 0.00 % | 0.0 |
| LRS-AKF (ours) | $n=4$, halo 48 | **0.0** | 0.012 | 0.00 % | 0.00 % | 0.0 |

Four things in this table are worth separating.

*Class A behaves exactly as Proposition 1 says it should.* The fixed-radius filter is *machine-exact* at a 48-pixel halo (RMSE and worst-pixel deviation both literally 0.0) because that halo exceeds its support, and at halo 16 its largest deviation is 0.12 of one 8-bit level, below the quantisation step. The grid-line visibility ratio nevertheless identifies what is left at halo 16 as a border effect: 255.8 at $n=4$ on this canvas (the $384\times384$ figure on the same $n=4$ grid is 14.2× (Section 6.2)) i.e. the residual error energy is 256× more concentrated on tile boundaries than on a random pixel set. The diagnostic names a seam that is *there* in float and *gone* after 8-bit quantisation, the case a "looks fine" inspection misses and a level-threshold criterion alone ignores.

*Class B does not behave, at four times the linear scale of the main experiment.* CA-AKF still differs from its whole-canvas result on 4.57 % ($n=2$) and 5.08 % ($n=4$) of the canvas at a 48-pixel halo (4.87 % and 5.65 % at halo 16) with 13.5 and 25.3 display levels at the worst pixel, and SC-AKF on 14.22 % and 24.10 %, with worst-pixel deviations of 123.1 and 145.3 levels at that halo. The plateau measured on $384\times384$ images (6.5 % and 20.5 % at halo 32, Table 4) is therefore not an artefact of the small images: it survives onto a canvas where the tiles themselves are 512–1024 px.

*Our operator is exact at the same scale.* LRS-AKF reaches literal zero RMSE at a 48-pixel halo on the canvas, with a worst-pixel deviation of $4.7\times10^{-5}$ (0.012 of one 8-bit level). At halo 16 (deliberately *below* the predicted halo $\rho+B = 24$ px) only 4 of 4.2 M pixels exceed one level and the worst is 4.9 levels, so the residual class-A truncation is present but bounded. We do not read this row's GV ratio of 0.0 as evidence about *where* the deviation sits: a four-pixel support is too small for the ratio to mean anything, and a shortfall of 8 px displaces a class-A residual into a band just inside each border rather than onto the border line itself. The difference between "a bounded band that disappears at a computable halo" and "a fixed fraction of the image that does not disappear at any halo" is the paper's entire taxonomy, and it is visible on a deployment-sized canvas, not only a $384\times384$ thumbnail.

Figure 5 shows the same configuration visually: whole-canvas crop, tile-wise crop, and the deviation amplified 40× on a shared scale, so that a sub-level deviation and a 25-level deviation are not made to look equivalent by per-panel normalisation.

*More overlap is not a dial that runs towards correctness.* Going from halo 16 to halo 48 reduces the class-B deviation by 10 % (CA-AKF, $n=4$: 5.65 % → 5.08 %) and 18 % (SC-AKF: 29.49 % → 24.10 %), while the worst single pixel of SC-AKF moves 175 → 145 levels and never approaches zero. Worse, the deviation *grows* when the tiles get smaller at fixed halo ($n=2 \to n=4$: CA-AKF 4.57 % → 5.08 %, SC-AKF 14.22 % → 24.10 %): a pipeline that switches to finer tiling to save memory changes the picture it renders, and no overlap setting common to both exists that prevents it.

### 6.7.1 A single artwork at native pixel density

The two canvases above are *assembled* from files whose long edge was normalised to 1024 px, so every photograph and every painting entered the mosaic down-sampled. That is a gap a reviewer is entitled to press on: the memory and partition mechanics of Section 6.7 do not care where the pixels came from, but the *content statistics* of a brushstroke at full resolution are a different regime: features that occupied two pixels in the served file occupy eight here, and the local window of the operator sees correspondingly more structure. This subsection closes the gap on one artwork at its scanned resolution; Figure 6 shows the memory comparison and the native-density ladder.

**Stimulus and provenance.** Metropolitan Museum of Art object 436104, *The Adoration of the Magi* (oil on wood, ca. 1520; European Paintings; CC0), served at its full scan size of $4000\times3853$ px, 15.41 Mpx, 184.9 MB as a float32 RGB array. Two derivatives are used, both produced by `make_native.py` from the retrieved JPEG and both recorded in `abstrabench/images_native/provenance.json` with their SHA-1, the SHA-1 of the source bytes, the crop offset and the derivation string: a lossless PNG that re-encodes exactly the pixels `cv2.imread` returns (asserted byte-equal at build time), and a centred $2048\times2048$ crop of those same pixels (offset $[902, 976]$). The crop is the informative one: it is *the same canvas size as the mosaic of Section 6.7*, so the difference between the two experiments is content density alone, not scale.

**Feasibility first.** On the full canvas the whole-canvas pass does not exist. All three methods abort identically (AKF, CA-AKF and LRS-AKF alike) with `MemoryError: Unable to allocate 2.07 GiB for an array with shape (12, 3853, 4000, 3) and data type float32`, having already reached 2798 MB of resident memory under a 5.5 GiB address-space cap on this 8 GiB host. The refused allocation is not an accident of our bookkeeping: it is the twelve-orientation rotated-Gaussian bank that the coherence-graded presmoothing step materialises in one `np.stack`, i.e. twelve canvas-sized float32 copies of the painting: the price of grading orientation in bins rather than per pixel. The tile-wise passes complete, at 1045–1122 MB (AKF), 1049–1123 MB (CA-AKF) and 1470–1629 MB (LRS-AKF) for halos 32 and 64 on a $4\times4$ grid, because a tile's bank is twelve copies of the *tile*. So the situation the paper predicts for real deployment is reproduced on a genuine large image rather than on an assembled one: *the reference is unavailable exactly where the pipeline has to run.*

That makes the Cauchy variant of Section 6.8 the operative instrument here, and it works. Running the same partition at halo 32 and at halo 64 and requiring the two outputs to agree certifies exactness without a reference. AKF returns **literally zero** differing pixels (RMSE 0.0, $\max|d|$ 0.0, GV 0.0). LRS-AKF returns 0 pixels above one quantisation level with $\max|d| = 5\times10^{-4}$ levels and GV 7.4: a residual two thousand times below the display step and still border-localised, which is the class-A signature. CA-AKF returns 422 differing pixels with a worst deviation of 1.49 levels and GV 0.98, flagged, interior, and not removable by the extra 32 px of overlap.

**Does the taxonomy survive native density?** On the $2048\times2048$ crop both passes exist, so the TCE ladder can be measured against the whole-canvas reference exactly as in Section 6.1 ($4\times4$ grid, $512\times512$ tiles, halos 16/32/48, one mode per process):

**Table 12.** TCE ladder against the whole-canvas reference on the centred $2048\times2048$ crop of the native $4000\times3853$ scan of Met object 436104 ($4\times4$ grid, $512\times512$ tiles, one mode per process) for AKF, CA-AKF and LRS-AKF (ours) at halos 16, 32 and 48 px: RMSE and worst-pixel deviation in 8-bit levels, fraction of pixels differing by more than one level and their count out of 4.19 M, grid-line visibility ratio and peak RSS (MB).

| method | halo | RMSE (levels) | worst pixel (levels) | frac $>1$ level | px of 4.19 M | GV ratio | peak RSS |
|--------------|-----|-------------|--------------------|-------------|------------|--------|--------|
| AKF | 16 | 0.0045 | 5.90 | 0.00029 % | 12 | 200.7 | 417 MB |
| AKF | 32 | **0.0** | **0.00** | **0.00 %** | **0** | 0.0 | 446 MB |
| AKF | 48 | **0.0** | **0.00** | **0.00 %** | **0** | 0.0 | 475 MB |
| CA-AKF | 16 | 0.418 | 17.03 | 4.96 % | 207 962 | 1.65 | 412 MB |
| CA-AKF | 32 | 0.410 | 16.02 | 4.88 % | 204 625 | 1.61 | 446 MB |
| CA-AKF | 48 | 0.392 | 14.93 | 4.64 % | 194 433 | 1.54 | 493 MB |
| LRS-AKF (ours) | 16 | 0.0111 | 13.67 | 0.00081 % | 34 | 82.2 | 543 MB |
| LRS-AKF (ours) | 32 | **0.0** | 0.00014 | **0.00 %** | **0** | 0.84 | 580 MB |
| LRS-AKF (ours) | 48 | **0.0** | 0.00025 | **0.00 %** | **0** | 0.57 | 612 MB |

Whole-canvas references on the same pixels: AKF 38.1 s / 2175 MB, CA-AKF 105.8 s / 2176 MB, LRS-AKF 270.3 s / 3894 MB.

Three readings, in the order a sceptic would ask for them.

*The class-B plateau is not an artefact of down-sampling.* CA-AKF is wrong on 4.6–5.0 % of a native-resolution Old Master painting at every halo we can afford, with worst-pixel deviations of 15–17 display levels and GV 1.5–1.7 (the error lives in tile interiors). That is the same plateau as on the assembled mosaic at the same grid and halo (5.08 % at halo 48, Section 6.7) and the same order as at $384\times384$ (6.5 % at halo 32, Table 4), reached with 194 433 pixels of brushwork changed by more than a quantisation level. Doubling the overlap from 16 to 48 px buys 6.5 % fewer wrong pixels and nothing else.

*The bounded-support operators are bit-exact on native pixels too, and the exactness is at the same halo as the theory predicts.* AKF goes from 12 pixels out of 4.19 M at halo 16 to *zero* at halo 32 (Table 12); LRS-AKF from 34 pixels (worst 13.7 levels, GV 82, a border band) to zero pixels, with a residual of $1.4\times10^{-4}$ levels at halo 32 that is float noise, not a different image. Nothing about native content density moved the exact halo.

*The Cauchy test detects class B but must not be read as an estimator of the error.* On the crop, where both numbers are available, the CA-AKF Cauchy pair at halos (16, 32) differs on 534 px and at (32, 48) on 1391 px (0.013 % and 0.033 %) while the same configurations differ from the whole-canvas reference on 4.96 % and 4.88 %. Two tiled runs of a global-statistic operator agree with each other 140× to 390× more closely than either agrees with the answer the deployment is supposed to reproduce, and the Cauchy figure is itself non-monotone in the halo (534 → 1391 px, the same effect as the 10/144 and 27/144 non-monotones of Section 6.8.3). The correct use of the test is therefore as a *detector* (a non-vanishing Cauchy pair is proof that no reference-free exactness claim can be made) and not as a bound on the deviation from whole-canvas processing. This is a qualification of our own instrument and we state it rather than leaving it for a reviewer to find.

*What this subsection does not do.* It is one artwork, one crop, one tile grid. It carries no statistical claim and no quality comparison, and its wall-clock numbers are not comparable to Section 6.7's: the tile-wise passes here ran concurrently with other experiments on a four-core host, showing a 2.8–3.2× overhead rather than the 1.36–1.48× measured on an unloaded machine. The memory ratio (5.2× for AKF, 4.9× for CA-AKF, 6.4× for LRS-AKF at halo 16) is the load-insensitive quantity the argument uses. A second native artwork ($3630\times4000$, `met_0436140`, CC0) is released with identical provenance but is not evaluated here.

## 6.8 Breadth: the consistency claim on 144 images across three content types

Sections 6.1–6.5 use 10 images because they record the full battery. This section re-measures the *central* claim (the tile-consistency behaviour, not the perceptual quality of the outputs) on a set an order of magnitude larger, and is a separate section rather than an extension of Table 4 for a deliberate trade: at $N=144$ images the full battery would be $144\times6$ metrics $\approx$ 12 h of single-threaded compute and would invite exactly the multiplicity criticism a large $n$ is supposed to answer. So one endpoint is pre-specified, *the fraction of pixels differing by more than one 8-bit level at halo 32, the most generous overlap we test*, and everything else (halo ladder, RMSE, exactness counts, grid-line concentration, non-monotonicity) is reported descriptively. The three paired comparisons (CA-AKF vs LRS-AKF, SC-AKF vs LRS-AKF, AKF vs LRS-AKF) form their own pre-specified family of three, corrected by Holm step-down, run on the Met pool; they are not added to the two families of Section 6.4.

**Scope, stated precisely.** $N=144$ of the 286 images available to us: 112 Met records from four departments (European Paintings, Photographs, Costume Institute, Islamic Art) *plus* every bundled scikit-image reference (10) and every synthetic phantom (22). The Met pool stopped at a cutoff of 110 distinct images, which the streaming ladder overshot by two. This is not "every image in the benchmark", and the acquisition ceiling behind it is in Section 7.

**Protocol.** Identical to Section 6.1 for the tiled pass ($4\times4$ tile grid, halos 0/16/32 px, $384\times384$ working size, one whole-image reference per method per image); 576 $(\text{image},\text{method})$ records, deduplicated and re-derived independently (reconciliation PASS on all 4 rows × 8 cells). Measured per-image runtime, whole + three tiled passes, one thread: AKF 13.1 s, CA-AKF 36.5 s, SC-AKF 94.8 s, LRS-AKF 94.9 s (medians). Comparisons are two-sided Wilcoxon signed-rank with `zero_method="wilcox"`, Holm step-down within each family (convention and corrected values in Appendix A).

### 6.8.1 Primary endpoint, pre-specified Met pool ($n=112$)

**Table 13.** Pre-specified primary endpoint on the Met pool ($n=112$): median fraction of pixels differing by more than one 8-bit level at halo 32 for each pair (A vs B) of CA-AKF, SC-AKF, LRS-AKF (ours) and AKF, with the counts of images where A $>$ B and A $<$ B, ties, the raw two-sided Wilcoxon $p$ and its Holm-adjusted value within the family of three (CA-AKF vs SC-AKF forming a separate family of one).

| pair (A vs B) | median A | median B | A $>$ B | A $<$ B | ties | raw $p$ | Holm $p$ |
|--------------------------|--------|--------|-------|------|-----|--------|--------|
| CA-AKF vs LRS-AKF | 6.192 % | 0.000 % | 112/112 | 0 | 0 | 4.10e-20 | **1.23e-19** |
| SC-AKF vs LRS-AKF | 12.766 % | 0.000 % | 106/112 | 1 | 5 | 2.81e-19 | **5.62e-19** |
| AKF vs LRS-AKF | 0.000 % | 0.000 % | 0/112 | 2 | 110 | 0.157 | 0.157 |
| CA-AKF vs SC-AKF (separate family of one) | 6.192 % | 12.766 % | 43/112 | 69/112 | 0 | 1.15e-07 | 1.15e-07 |

The third row of Table 13 is a *null by construction*: both bounded-support operators already sit at the quantisation floor, so the endpoint cannot separate them, which is the point, and the reason the fixed-radius control belongs in the family rather than being reported as a win. Re-running the same four tests over all 144 images gives the same interpretation with different magnitudes (CA-AKF vs LRS-AKF 137/144, $p=3.15\times10^{-24}$; SC-AKF vs LRS-AKF 123/144, $p=4.49\times10^{-22}$; CA-AKF vs SC-AKF 61/144, $p=2.38\times10^{-5}$; AKF vs LRS-AKF null at $p=0.083$); the Met-only numbers are the pre-specified ones and the 144-image set is the replication.

### 6.8.2 Descriptive table by content type

**Table 14.** Descriptive halo ladder by content stratum on the Section 6.8 protocol ($4\times4$ tile grid, $384\times384$ working size) for CA-AKF, SC-AKF, LRS-AKF (ours) and AKF (control): median fraction of pixels differing by more than one 8-bit level at halos 0, 16 and 32 px, median RMSE at halo 32, and the number of images exact by halo 32 and already exact at halo 16, over Met records (112), bundled scikit-image references (10) and synthetic phantoms (22).

| stratum | method | median frac $>1$ level, h0 / h16 / h32 | median RMSE h32 | exact by h32 | exact already at h16 |
|-----------------|--------------|--------------------------|---------------|------------|--------------------|
| Met (112) | CA-AKF | 14.57 % / 7.63 % / 6.19 % | 0.002603 | 0/112 | 0/112 |
| Met (112) | SC-AKF | 23.20 % / 14.66 % / 12.77 % | 0.017614 | 6/112 | 5/112 |
| Met (112) | LRS-AKF (ours) | 7.78 % / 0.00 % / 0.00 % | 0.000000 | 110/112 | 1/112 |
| Met (112) | AKF (control) | 7.92 % / 0.00 % / 0.00 % | 0.000000 | 112/112 | 4/112 |
| skimage refs (10) | CA-AKF | 15.96 % / 11.14 % / 8.99 % | 0.002944 | 0/10 | 0/10 |
| skimage refs (10) | SC-AKF | 17.02 % / 6.80 % / 5.81 % | 0.008565 | 3/10 | 0/10 |
| skimage refs (10) | LRS-AKF | 6.22 % / 0.00 % / 0.00 % | 0.000000 | 10/10 | 1/10 |
| skimage refs (10) | AKF | 7.14 % / 0.00 % / 0.00 % | 0.000000 | 10/10 | 2/10 |
| synthetic (22) | CA-AKF | 10.87 % / 1.35 % / 1.31 % | 0.001555 | 7/22 | 7/22 |
| synthetic (22) | SC-AKF | 6.10 % / 0.02 % / 0.00 % | 0.000007 | 11/22 | 9/22 |
| synthetic (22) | LRS-AKF | 6.09 % / 0.00 % / 0.00 % | 0.000000 | 21/22 | 11/22 |
| synthetic (22) | AKF | 4.89 % / 0.00 % / 0.00 % | 0.000000 | 22/22 | 15/22 |

The Abstract's and Section 9's $6.19\%$ and $12.77\%$ are these Met-stratum ($n=112$) medians at halo 32 (Table 14); pooled over all 144 images the same medians are 5.51 % (CA-AKF) and 8.37 % (SC-AKF). Absolute levels are not comparable across strata: every method is lower on synthetic content, since the phantoms are smoother and a flat image makes a global quantile nearly tile-invariant. What carries is within-stratum ordering, preserved in all three strata. Figure 7 plots the halo ladder per stratum.

### 6.8.3 Four things that survive the change of scale

*The two classes separate the same way, and the separation is not a content-type artefact.* On the Met pool the local operator is exactly reproducible tile-wise at halo 32 on 110/112 images and the class-B references on 0/112 (CA-AKF) and 6/112 (SC-AKF); on the bundled references the split is 10/10 versus 0/10 and 3/10; on the phantoms 21/22 versus 7/22 and 11/22. The honest generalisation is therefore **"no finite halo *guarantees* exactness for a class-B operator"**, not "class B never achieves it": when the global statistic happens to be insensitive to cropping (seven synthetic phantoms, six Met images for SC-AKF) the tiled and whole passes do coincide at the 8-bit floor. What is not observed anywhere in 576 records is a class-B operator whose deviation *converges to zero as a function of the halo*: the plateau is a property of the statistic, and whether it lands above or below one display level on a given image is a property of the content.

*The halo ladder is not monotone for class B, and that is a measurable failure mode.* Counting images whose halo-32 deviation is *larger* than their halo-16 deviation: CA-AKF 10/144, SC-AKF 27/144, against 0/144 for both bounded-support operators. Paying for twice the overlap makes the picture *worse* on one image in five for SC-AKF. A pipeline tuning its overlap by inspection has no monotone objective to tune against, which is a stronger practical statement than the plateau itself.

*Most of the deviation is not a seam.* Grid-line visibility at halo 16, restricted to images whose deviation is large enough for the ratio to mean anything ($>0.1$ % of pixels above one level), has median 1.0 over 112 Met images for CA-AKF and 1.1 over 103 for SC-AKF: the deviation energy is no more concentrated on tile boundaries than on a random pixel set. Concretely, 124/137 CA-AKF and 107/119 SC-AKF contributing images sit below $2\times$; only 1 and 3 images respectively exceed $4\times$ (`syn-checker-24px.png`; and for SC-AKF `met_0100032.jpg` 6.3, `met_0259605.jpg` 9.1, `syn-checker-24px.png` 30.7). We do not report GV medians for the two bounded-support operators at this scale: their h16 deviation clears the gate on a single image (`syn-checker-24px.png`, 26.3 and 30.7), so the corresponding AKF figures (14.2× at $384\times384$, Section 6.2; 255.8× on the $2048^2$ canvas, Section 6.7) rest on one contributor and are not method statistics. No blending window, seamline search or per-boundary overlap schedule can remove error that does not live on the boundaries.

*Proposition 1 is confirmed in the qualitative form and falsified in the sharp form, and we report which is which.* The sharp reading says a locally-support-$B$ operator is exact at halo $B$, and for the fixed-radius control $B$ is often quoted as $2r+\rho$ (with $r=5$, $\rho=1$, $B=11$ px, so a ladder {0,16,32} should give exact at 16). Observed: AKF is exact at halo 16 on 21/144 images and needs halo 32 on 123/144, with the mismatch strictly one-sided (never exact *earlier* than predicted). The synthetic phantoms localise why: every clean locally-periodic pattern (checker-8px/64px, 1-px lines, chirp stripes, radial and spiral flow, colour-edge quads, junction-plus, T-brick) matches the $2r+\rho$ prediction, while all stochastic or high-frequency-texture images, the three `disc-noise` phantoms, `lines-random-thin`, `checker-24px`, `junction-x-2px`, `t-junction-segments` and 8/10 photographs, need 32. The reason is in the implementation, not the theory: this AKF configuration still contains unbounded-support Gaussian smoothing (`sigma_along=2.5`, structure-tensor `sigma_t=4.0`, `sigma_g=1.0`), so its true footprint exceeds $2r+\rho$ and the *radius* is not the quantity to plug in. The qualitative reading (bounded support implies exactness at *some* finite halo) holds on 144/144 for AKF. For LRS-AKF the two nominal footings of the same chain, $11+\rho=19$ px ($2r+\rho_{\text{base}}$) and $B+\rho=16+8=24$ px (Section 6.6's), agree on both ladders: 32 px on this $\{0,16,32\}$ ladder, 24 px on Section 6.6's 8-px ladder, where 4/4 images are exact. On the 144-image ladder 128 land on the prediction, 13 turn exact one rung earlier (permitted for a sufficient condition) and 3 are never exact (Section 6.8.4 names them), so we quote measured exact haloes throughout and give Section 8's design rule the caveat that $B$ must be measured for the implementation in hand.

### 6.8.4 The weakest links, named

Three of 576 records leave the local operator not exact at the largest halo tested. Each is *one pixel out of 147,456* (`frac`$=6.782\times10^{-6}$ exactly), with $\max|d|$ of 1.30 levels (`met_0259569.jpg`), 3.32 levels (`met_0259599.jpg`) and 1.22 levels (`syn-t-junction-segments.png`); the ladder still improves monotonically on all three (e.g. 11 px $\to$ 1 px for `met_0259599`). These are the images where the claim is weakest and they are in the release by name, not aggregated away. For comparison, the class-B median at the same halo on the 112 Met records is 6.19 % (CA-AKF) and 12.77 % (SC-AKF) of pixels wrong, with median worst-pixel deviations of 15.2 and 111.7 levels against our 1.2–3.3, one to two orders of magnitude larger.

What this section does *not* establish: any quality comparison at this $N$ (Sections 6.4–6.5 are the quality evidence, at $n=10$), and anything about domains outside museum records, bundled reference images and deterministic phantoms (Section 7). The stop point of 110 distinct Met images was fixed before aggregation, when the marginal image had become another record from the same four departments rather than a new content type; the run is resumable, the two-image overshoot is a polling artefact and not a selection, and the released records cover every image processed.

## 6.9 Published algorithms, third-party code

Every operator measured so far in this paper is one we wrote, which leaves the obvious
suspicion that the class-A/class-B taxonomy describes *our code* rather than the operators.
To our knowledge no prior study reports a tiled-versus-whole-image consistency *measurement*
for a published non-learned filter. The tiling-consistency literature is entirely about trained
networks, receptive field and per-tile feature statistics ([@reina2020tiling; @huang2018tiling; @buglakova2025tiling]), and for linear shift-invariant filters the
question is vacuous, since overlap-add is exact by construction; the classical analogue that
does exist is shift/translation invariance of the whole-image estimate, which moves the sampling
grid rather than partitioning the image. That is the gap this section fills: it runs the
identical protocol, $384\times384$ working size, $4\times4$ tile grid, whole-image reference,
the exactness predicate $\max|d| < 1/255$ *and* zero pixels above one level, against seven
**published algorithms as implemented by other people**, shipped in OpenCV 5.0.0 and
opencv_contrib 5.0.0, on the same 144 benchmark images (1008
$(\text{image},\text{method})$ records); Figure 8 shows the resulting ladder and the determinism control. The searches behind the absence-of-evidence claim
above, with DOIs and query strings, are released with the paper (Appendix A). Three differences from Sections 6.1–6.8
are deliberate and are recorded per row rather than smoothed over.

*These functions take and return 8-bit data*, so one 8-bit level is the native resolution of
the method, not a display convention borrowed from a float pipeline: the exactness predicate
here is an integer statement about the output array.

*One setting per method, chosen before the ladder was measured*, taken from the library's own
documentation or the paper's recommended value, and *a support bound registered for each* before measurement: bilateral `d=9` → exact at halo 5; guided `radius=16` → 32
(box averages applied twice); NLM template 7 / search 21 → 30; Perona–Malik 20 iterations →
20 (one pixel of stencil propagation per iteration); adaptive manifolds $\sigma_s = 9$ → 36;
rolling guidance and L0 smoothing → **no bound predicted**, because both contain a
whole-image solve. The prediction is therefore auditable against the measurement, and a
method that beats its bound is reported as beating it.

*A determinism control is added*, because the protocol assumes throughout that the operator is
a function of pixel content: the same array is filtered twice, and identical content is placed
in a fresh allocation and in a non-contiguous view of a larger buffer.

"n/a" in the GV column of Table 15 means the method is already exact to within the gate used in
Section 6.8.3 (more than 0.1 % of pixels differing) at halo 16, so the ratio is not defined,
which is itself the class-A outcome. The median exact halo is taken over **all** 144 images of
a method, with an image that never becomes exact inside the ladder censored at "not exact at 48";
taking the median over the converging images only would report 32 px for adaptive manifolds and
0 px for L0 smoothing, which is not what those operators do.

**Table 15.** Three bounded-support algorithms become bit-exact inside their pre-registered halo on all 144 images, while L0 smoothing stays wrong on 141 of 144 at halo 48. Seven published algorithms as shipped in OpenCV 5.0.0 and opencv_contrib 5.0.0 were measured on the same 144 images at $384\times384$ with a $4\times4$ tile grid, each with one setting and one support bound registered before its ladder was run. Columns give the predicted exact halo, the median measured exact halo, images exact within the prediction, images never exact at halo 48, the fraction of pixels differing by more than one 8-bit level at halo 0 and at halo 48, the worst-pixel deviation in levels at halo 48, and grid-line visibility at halo 16.

| method | refs | setting | predicted exact halo | median measured | within prediction | never exact at 48 | frac $>1$ lv at halo 0 → at 48 | worst px at 48 | GV at 16 |
|------------------|--------------------------|--------------------------|--------------------|---------------|-----------------|-----------------|--------------------------|--------------|--------|
| bilateral | [@tomasi1998bilateral] | d=9, sColor=75, sSpace=75 | 5 px | 4 px | 144/144 | 0/144 | 3.32 % → 0.000 % | 0.0 lv | n/a |
| guided | [@he2010guided] | radius=16, eps=(0.1*255)^2 | 32 px | 32 px | 144/144 | 0/144 | 18.44 % → 0.000 % | 0.0 lv | 6.72 |
| NLM | [@buades2005nlm] | h=10, hColor=10, template=7, search=21 | 30 px | 16 px | 144/144 | 0/144 | 12.03 % → 0.000 % | 0.0 lv | n/a |
| Perona–Malik | [@perona1991diffusion] | alpha=0.5, K=0.2, iters=20 | 20 px | not exact at 48 | 0/144 | 144/144 | 51.57 % → 14.758 % | 193.5 lv | 0.57 |
| adaptive manifolds | [@gastal2012amf] | sigma_s=9, sigma_r=0.1 | 36 px | not exact at 48 | 8/144 | 132/144 | 12.71 % → 0.015 % | 3.0 lv | 3.58 |
| rolling guidance | [@zhang2014rollingguidance] | d=-1, sigmaColor=35, sigmaSpace=15, iters=3 | none (global solve) | 48 px | n/a | 26/144 | 13.75 % → 0.000 % | 1.0 lv | 6.25 |
| L0 smoothing | [@xu2011l0smooth] | lambda=0.02, gradient=0.7 | none (global solve) | not exact at 48 | n/a | 141/144 | 99.95 % → 99.877 % | 208.5 lv | 1.00 |

For comparison, the same 144 images under our own four operators at the Section 6.8
protocol: AKF exactly reproducible tile-wise at halo 32 on 144/144, LRS-AKF (ours)
on 141/144, SC-AKF on 20/144 and CA-AKF on 7/144.

**Class A is the norm, and the pre-registered bounds hold.** All three bounded-support
methods are exactly reproducible tile-wise within the ladder, and on every image the measured
exact halo is *at or inside* the registered prediction: bilateral 144/144, guided
144/144, NLM 144/144. NLM is the instructive case: predicted 30, measured median
16 px: the bound is sufficient, not tight, and the implementation's prefiltering makes
it converge earlier than the naive stencil count suggests. Exactness at halo 0 is rare and
structured (0/144, 3/144, 2/144 images, and every one of them a
synthetic phantom whose pattern period divides the tile size, so the partition is invisible to
the operator), which is the same "tiling is never free" observation as Section 6.1. (Citation
note, since we are naming other
people's code: OpenCV's own headers declare no reference for `bilateralFilter` at all, cite the
AVSS 2005 companion paper for the video variant, and link the IPOL re-publication for the
denoiser; we cite the primary algorithm papers, which is what the measured behaviour should be
attributed to.)

**Two published operators have support that is not bounded by a constant, and the ladder shows
it.** `amFilter` is a two-pass recursive IIR filter along scanlines: its deviation decays with
overlap (12.71 % → 0.015 %) but its exact halo exceeds the registered 36 px on
132/144 images even at 48 px, with GV 3.58, a border effect, but one whose width is
set by the image, not by the parameter. `rollingGuidanceFilter` behaves the same way more
mildly (26/144 not exact at 48). This is the practical content of the
design rule in Section 8 stated on foreign code: *the halo has to be measured for the
implementation in hand*, because the number written in the method section of the paper is not
the footprint of the code.

**Class B is real in third-party code, and it is worse than our reference implementations.**
`l0Smooth` solves a global gradient-domain system, and it shows: not exactly reproducible on
141/144 images at *any* halo in the ladder (the three exceptions are again periodic
phantoms, `syn-junction-plus-1px`, `syn-lines-2px-pair` and `syn-t-junction-brick`, where the
partition costs the operator nothing), with 99.877 % of pixels differing by
more than one level at halo 48 and a worst pixel of 208.5 lv, and GV 1.00, i.e. the
disagreement is spread through tile interiors, exactly the class-B signature. Perona–Malik
diffusion plateaus at 14.758 % at halo 48 with worst pixel 193.5 lv and GV 0.57.
Neither can be repaired by overlap, and neither was designed with tiling in mind, which is the
point: the taxonomy is a property of the parameter chain, not of the authors' intentions. Two
honest qualifications on the second case, both read off the shipped source: OpenCV's
`anisotropicDiffusion` is the *scalar* Perona–Malik equation (their Eq. 1) with the exponential
conductivity only, an 8-neighbour explicit Euler step with a `saturate_cast` to `uchar` after
every iteration, so the "second, improved" conductivity documented in the header is not
implemented, and the diffusion process is therefore not exactly the published algorithm; and
its non-locality is in any case the expected class-A behaviour of 20 accumulating iterations if
the stencil were applied consistently, which is why the determinism result below, not this
plateau, is the load-bearing finding for Perona–Malik.

**A third failure mode, outside the taxonomy.** The determinism control was meant as a
sanity check. On 40 images probed, six of the seven implementations return bit-identical
output for identical content in all three variants. `anisotropicDiffusion` does not: called
twice on the *same unchanged array* it returns different outputs on 40/40 images
(median 262 pixels differing, worst 255 of 255 levels), and the same holds for a
fresh allocation with identical content (40/40) and a non-contiguous view of the
same values (40/40). The three probes are not equally severe. A repeat call moves
a median of 262 pixels while the same content in a fresh allocation moves
6,476, which is what a dependence on uninitialised memory rather than on scheduling
looks like. One iteration is bit-exact; two or more are not. The
candidate mechanism is visible in the shipped source
(`modules/ximgproc/src/anisodiff.cpp:274-288`): from the second iteration the border
replication writes into a buffer that also *holds* the view being replicated, and the
1-pixel frame of the alternate buffer is never initialised. We report this as a measured,
reproducible property of a released version, not as a diagnosis: the two memory defects behind
it have been on record in an open upstream issue since 2018 (`opencv/opencv_contrib#1573`)
without being connected to run-to-run variation, and a downstream report of the variation
itself (`emgucv/emgucv#860`) was closed as not reproducible with an attribution, thread-count
dependent floating-point accumulation, that our control refutes, since the divergence persists
identically with `cv2.setNumThreads(1)` and with 4. We do not claim to be the first to observe
it, and we have not filed a new upstream report.

**What this section does and does not establish.** It establishes that the two classes are
observable, separable and correctly predicted in published code we did not write, on a
benchmark of 144 images, with per-method bounds registered in advance, which converts the
paper's central claim from a description of our implementations into a statement about
operators. It does *not* establish any quality ranking: these seven are denoisers and
smoothers, not abstraction filters, and none of them was proposed as a Kuwahara variant; the
comparison in Sections 6.4–6.5 remains against the adaptive-Kuwahara family, where it belongs.
It also inherits the version pin of every such study: the numbers above are properties of
OpenCV 5.0.0 / opencv_contrib 5.0.0, and the source lines are quoted so that a reader can check
them against that tag rather than against our prose.


## 6.10 An audit of distributed Kuwahara implementations

Section 6.9 establishes that the two classes are observable in published code, but none of its seven operators is a Kuwahara variant. A reader can therefore still ask whether the pattern we name occurs in the adaptive-Kuwahara implementations that are actually distributed. This section answers that question by reading the expression that forms the parameter in every such implementation we could obtain in source form, and recording the domain over which each expression reduces the image. It is a source audit, not a re-run: three of the four are Qt/GLSL or TouchDesigner projects that we could not build on this host, and a build we did not run is not reported as a measurement. One further candidate archive could not be read at all (its distribution endpoint returned an HTTP 404 body), so it is excluded rather than counted.

**Table 16.** Four distributed anisotropic-Kuwahara implementations, read at the expression that forms the parameter. The reduction domain is the set of pixels whose values enter that expression. Three trees were read as source; the fourth ships only compiled operator files. Exact line ranges are recorded in Appendix A.

| implementation and relation | source read | expression that forms the parameter | reduction domain | class |
|--------------------------|--------------------------|--------------------------|--------------------------|------------|
| `gpuakf` (2009), independent | `akf_v2.glsl`, `tfm.glsl`, `glview.cpp` | sector variance $\sigma^2$; weight $1/(1+(255\sigma^2)^{q/2})$; anisotropy from the local structure tensor. `radius`, `q`, `alpha` are UI uniforms | bounded by twice `radius` | A |
| `polyakf` (2010), independent | `akf_v2n4.glsl`, `polyakf.cpp` | polynomial weight $1/(1+(255\sigma)^{q})$ from sector standard deviations; uniform-only parameters | bounded | A |
| TD-Anisotropic-Kuwahara (2017), independent | `README.md`, four `.tox` binaries | declares a port of `gpuakf`; ships no readable shader text and reports kernel generation as unfinished | inherited by declaration, not verified | A (declared) |
| JLR-CAKF operator base, predecessor project imported by our two references | `filters.py`, `repair.py`, `methods.py` | sector scale is a per-pixel median across $K$ sectors; the structure-repair line mask is thresholded at an image-wide 99th percentile | bounded on the support path, whole-image on the repair branch | A / B |

Three findings follow. First, on the main support-selection path, none of the four uses a whole-image reduction. The published implementations are class-A by construction, and Proposition 1 describes what they already do; our two class-B references are deliberately constructed variants, not recovered examples of a widespread bug. Second, the pattern is not confined to code we wrote: the structure-repair branch of the predecessor pipeline uses an image-wide 99th percentile as a mask threshold, and Section 6.9 finds two OpenCV algorithms with a global solve whose behaviour the class-B prediction describes. Third, the audit is bounded. Four implementations, one release of each, three of them read rather than executed, do not support a prevalence claim, and we withdraw the one made in earlier drafts of this paper.

The contribution is therefore stated as a correctness criterion, an instrument that detects a violation in code we did not write, and a measured consequence for reported numbers (Table 7). None of the three requires the violating pattern to be common. A reader who wants to know whether a given pipeline has the property can apply the protocol of Section 4 without relying on this audit, and a reader who wants to know how often deployed code has it should treat Table 16 as the beginning of that census, not its conclusion.


## 6.11 Independent replication on a second numeric stack

The primary tables were produced on a Linux host (Python 3.12.13). We rebuilt the same operator stack on Windows (Python 3.13.14, numpy 2.5.3, scipy 1.18.1, scikit-image 0.26.0, OpenCV 5.0.0), verified that the operator source files and the image hashes are identical, and re-ran the full battery on 23 images: the ten images of the primary cohort plus 13 new museum records. A same-image check does not reach bit-exactness across the two stacks (median relative drift 0.07 % for CA-AKF and 12 % for LRS-AKF, the latter on an RMSE of order $10^{-4}$), so this is reported as an independent replication on a second numeric stack, not as an extension of the primary quality table. All 23 images were run in one environment, so the statistics below are internally homogeneous.

**Table 17.** Quality family on the independent replication cohort of 23 images (ten shared with the primary cohort of Section 6.4, 13 new museum records): mean $\pm$ sample sd, paired mean difference $\Delta$, paired effect size $d_z$, a 95 % paired-bootstrap interval, images won, and the two-sided Wilcoxon $p$ with its Holm-adjusted value. All six paired differences keep the sign of Table 6.

| metric | CA-AKF | LRS-AKF | $\Delta$ | $d_z$ | 95 % CI of $\Delta$ | win/23 | $p$ | $p_{\mathrm{Holm}}$ |
|------|----------------|----------------|---------------|-----|------------------|------|-------|--------|
| SR | 0.5522 $\pm$ 0.096 | 0.5867 $\pm$ 0.093 | +0.0345 | +0.80 | [+0.0163, +0.0504] | 20/23 | 0.0024 | 0.0072 |
| SSIM | 0.7109 $\pm$ 0.107 | 0.7417 $\pm$ 0.104 | +0.0308 | +2.28 | [+0.0255, +0.0364] | 23/23 | 2.7e-05 | 0.00016 |
| FSR | 0.1875 $\pm$ 0.058 | 0.2157 $\pm$ 0.067 | +0.0281 (worse) | +1.10 | [+0.0197, +0.0398] | 1/23 | 3.1e-05 | 0.00016 |
| ECE | 19.3344 $\pm$ 6.072 | 18.4710 $\pm$ 5.939 | -0.8634 | -0.68 | [-1.3312, -0.3080] | 18/23 | 0.00085 | 0.0034 |
| SCG | 0.0929 $\pm$ 0.035 | 0.0999 $\pm$ 0.030 | +0.0070 | +0.61 | [+0.0023, +0.0116] | 16/23 | 0.0067 | 0.0134 |
| OVT | 0.0044 $\pm$ 0.005 | 0.0049 $\pm$ 0.004 | +0.0005 | +0.11 | [-0.0015, +0.0024] | 9/23 | 0.4379 | 0.4379 |

All six paired differences keep the sign of the primary table, and the smallest two-sided Wilcoxon $p$ is $2.7e-05$ rather than the $0.00195$ floor. Five of the six metrics survive Holm correction. Two conclusions change with the added power, and both are reported rather than smoothed over. SR, ECE and SCG, indistinguishable at $n=10$, are now significant in the same direction, so the conservative reading of the primary table understates those effects. OVT does the opposite: its $n=10$ bootstrap interval excluded zero while the rank test did not reject, and in the replication the interval is $[-0.0015, +0.0024]$ with $p=0.438$. The apparent OVT effect was a small-sample artefact, which is why both statistics are reported for every metric. The consistency claim also replicates: median RMSE at $n=8$, halo 16 is $3.70e-03$ for CA-AKF against $9.83e-05$ for LRS-AKF (paired $p=2.4e-07$), and the harness result is stronger than in the primary cohort: SC-AKF's ECE moves by $+1.22$ on average, while the two bounded-support operators move by at most $0.0032$ on any metric.

The cohort is deliberately not pooled with Table 6. Pooling would require the two stacks to agree bit-exactly, which the ten shared images show they do not; reporting the second stack separately keeps the platform effect visible and lets a reader check both. The unaggregated records, the pinned environment and the per-image drift on the shared images are released with the paper (Appendix A).


# 7 Threats to validity

*Threshold choice.* We use one 8-bit level as the "noticeable" unit. This is a convention, not a perceptual study; the class-A/class-B distinction does not depend on it, since the two cases differ in whether the error *vanishes* as the halo grows, not in how large it is at one threshold.

*Image set.* The released benchmark is 286 images: 254 CC0 records from The Metropolitan Museum of Art open collection (four departments, European Paintings 145, Photographs 44, Costume Institute 40, Islamic Art 25), 10 bundled public-domain reference images from the scikit-image data bundle, and 22 deterministic synthetic phantoms. The full metric battery (six quality metrics, border-band decomposition and grid-line visibility, measured per image and method) is reported on 10 images (Sections 6.1–6.5), which is the resolution floor for the paired tests we run ($p_{\min}=0.00195$ at $n=10$; Section 6 preamble). Section 6.4 reports paired effect sizes and bootstrap intervals for that family so that a floored $p$-value is not mistaken for a small effect. The central consistency claim is re-measured on 144 of those images: 112 museum records plus every bundled reference and every synthetic phantom (Section 6.8). Four limitations follow.

First, the breadth claim is a *single-endpoint* claim: Section 6.8 pre-specifies one primary quantity (fraction of pixels differing by more than one 8-bit level at the most generous halo) and reports everything else descriptively, so the large $n$ is not bought with additional testing.

Second, the pool is an *acquisition ceiling*, not a sampling design. The Met Open Access API returned HTTP 429 on repeated department paging and Cloudflare 403 once departments were probed in parallel, so 254 licence-verified records is what we could retrieve in the time available; Section 6.8's stop point was fixed before aggregation. The stratum is texture-rich but institutional (one museum, four departments) and photographic, printed and textile content enters both as real records (Photographs, Costume Institute, Islamic Art) and as synthetic phantoms.

Third, *served files are normalised to a longest side of 1024 px*, and the pool therefore contains no native-resolution image by design; four files above that size exist in the release by exception, two Metropolitan Museum originals at native resolution ($4000\times3853$ and $3630\times4000$) and their two $2048\times2048$ crops, listed with their SHA-1s, source hashes and crop offsets in `abstrabench/images_native/provenance.json`, and they are used only by Section 6.7.1. That subsection is a *single-artwork* experiment: the memory and feasibility behaviour is reproduced on genuine native pixels, the class-A/class-B split is reproduced there at the same halos, and nothing else is. It does not substitute for a large native-image benchmark, which the acquisition ceiling above makes impossible for us, and the content statistics of other 4000-px corpora (whole-slide pathology, satellite orthophotos, stitched driving panoramas) remain untested by us (fourth limitation below).

Fourth, *domain*. Every image here is art, photography, costume or a synthetic phantom. The design rule in Section 8 is stated for any operator whose parameter is normalised by a global statistic, which includes remote-sensing, histopathology and driving imagery pipelines; we have not run those domains, so we do not claim their validation, and the analogue we do draw (data leakage across shards in learned pipelines, Section 8) is labelled as an analogy rather than as evidence.

*Third-party evidence is version-pinned and single-setting.* Section 6.9 measures OpenCV 5.0.0 / opencv_contrib 5.0.0 at one documented setting per method; it is a statement about that release, not about the algorithms in general, and the source files and line ranges are quoted so a reader can re-check them against the tag. The pre-registered support bounds are our reading of each published algorithm, not the library's claim, and one of the seven implementations does not match the algorithm it documents (its second conductivity is declared but unimplemented), which is reported in Section 6.9 rather than quietly absorbed.

*Literature coverage.* Bibliographic verification ran through OpenAlex, Crossref and the museums' own APIs; query strings and per-lane hit lists are released with the paper (Appendix A). OpenAlex rate-limited the four most recent competitor lanes, so those were verified against primary pages (publisher or arXiv abstract) rather than through a bibliographic index, and five 2022–2026 records without an abstract in the index are cited on title-and-venue evidence only, flagged per entry in the released bibliography. The consequence is a narrower search than a systematic review would claim: the nearest mechanism-level prior art we found [@buglakova2025tiling], on tiled-evaluation artefacts attributed to per-tile feature-normalisation statistics in learned networks, is named and distinguished in Sections 2.2 and 8, and the four precedents added for adaptive histogram equalisation, tile-boundary coding, exact tile-based inference and streaming multigrid are distinguished in Section 2.2, but we cannot assert that nothing closer exists. No record was found that states our result: that a whole-image parameter functional makes exactness unattainable at any finite halo.

*Baselines.* CA-AKF and SC-AKF are reference *implementations of the design pattern*, not published methods: each builds the same anisotropic-Kuwahara pipeline (Section 5.3, identical pixel-space settings taken from the AKF literature [@kyprianidis2009akf; @kyprianidis2011msakf]) and differs only in that its threshold is a functional of the whole image: a 95th percentile of the structure-tensor eigenvalue in CA-AKF, a tuned quantile of the dispersion map in SC-AKF. We chose them because they are the two most natural ways this pattern appears in practice. Section 6.9 measures seven published algorithms in third-party code, where the taxonomy is confirmed on operators we did not write, and Section 6.10 audits the adaptive-Kuwahara implementations themselves. That audit found no whole-image reduction on their main support-selection path, which is why the paper makes no prevalence claim. We do not claim LRS-AKF is optimal within its own family; the claim is about consistency, which is configuration-independent.

*Generalisation.* We prove consistency for one operator and measure it for two reference implementations of the violating pattern. Section 8 states the design rule for the wider class. Section 6.9 audits seven published implementations of neighbouring operators at one pinned version and one documented setting each, and Section 6.10 reads the parameter expressions of four distributed Kuwahara implementations. Both are samples, not a census, and the design rule is stated as a condition on a parameter chain rather than as a claim about how frequently the violating chain is shipped.

# 8 Discussion: a design rule for adaptive filters

The finding generalises past this filter. A content-adaptive operator can be evaluated as independent tiles on a large canvas if and only if every quantity that can change a pixel's output is computable from a bounded neighbourhood of that pixel. In practice this means:

1. **Do not expect a whole-image parameter to survive partitioning.** If a parameter is a percentile, quantile, mean, or normalisation constant of the entire image, no finite halo restores exactness (Proposition 2). Replace it with a bounded-support local estimate, accepting a finite halo that you can then *prove* sufficient. The same test applies to auto-levels and to global histogram matching, where the transfer function is itself a whole-image functional. Adaptive histogram equalisation is the precedent that the replacement works in practice [@pizer1987clahe]; the point here is that the replacement needs a stated support bound and a measured halo, not only a smaller window.
2. **Prefer ratios to thresholds.** A dimensionless criterion ($v_{r_i}(x)/R(x)$) is automatically invariant to the amount and distribution of content elsewhere.
3. **Report tile consistency.** A filter that cannot be evaluated in tiles cannot be evaluated at scale, and a benchmark that only evaluates whole images will systematically over-report methods that smuggle in global statistics.
4. **Streaming summaries are not a fix.** A bounded-memory quantile estimator [@greenwald2001gk] makes a global statistic cheap; it does not make it local. A two-pass scheme that computes the statistic over the whole image and then filters in a second pass does reproduce the whole-image result, at the price of two full reads of the data and a barrier between them; the claim in this paper is about one-pass, tile-independent evaluation, which is what production pipelines do.
5. **Measure the halo; do not derive it.** Proposition 1 turns a *support* into a *sufficient halo*, and the two are not the same number in real code: for the fixed-radius control, $2r+\rho$ is met on 21/144 images while 123/144 need twice that halo, and the excess traces to Gaussian presmoothing and structure-tensor scales that never terminate. A pipeline that sizes its overlap from the window radius in the paper's method section is under-overlapping by construction. The cheap check is the Cauchy criterion used in Section 6.7.1: run the same partition twice at two halos and require the deviation to fall under one quantisation level.
6. **Do not tune overlap by trial.** For a class-B statistic the halo ladder is not monotone: $10/144$ images get *worse* from halo 16 to halo 32 for CA-AKF and $27/144$ for SC-AKF, against $0/144$ for both bounded-support operators, so "increase overlap until the seams disappear" has no monotone objective to follow, and the image it converges to (if any) depends on where the ladder happened to stop.
7. **Low-rank structure is not an escape hatch.** The tempting theoretical rescue (factor the second-moment matrix, shrink the state, make the tiled pass cheap and exact) does not exist here. The $2\times2$ PSD bound of Property 1 is a statement about the domain dimension, its rank-1 truncation error is exactly $\lambda_{\min}$, and measured $\lambda_{\min}/\lambda_{\max}$ exceeds $0.5$ on 17–71 % of pixels while exact rank deficiency occurs on 0.0 % (Section 3.1). Locality is the property that matters; conditioning is not available to buy it.

**Bounded memory is not bounded support.** Streaming multigrid computes a global gradient-domain solution out-of-core without partitioning the operator [@kazhdan2008smg], which shows that a global *computation* can be memory-bounded. It does not make the *parameter* local: the tiles remain coupled through the multigrid hierarchy, the result is still the global solution, and the method is specific to linear Poisson systems. L0 smoothing, the class-B case in Section 6.9, is a non-convex sparsity problem whose active set is data-dependent, so that route is not available to it. The distinction matters for the design rule: bounded memory and bounded support are different properties, and only the second survives an arbitrary partition.

**Where this lands outside graphics, stated as an analogy, not as evidence.** A parameter computed from the whole image and then applied as if it were local is the same structural error as fitting a normalisation statistic on data that will not be present at inference. In learned pipelines the instances are well known: train/test leakage through a full-corpus statistic, batch normalisation running differently per shard in a distributed trainer, and, closest to our case, tiled evaluation artefacts attributed to per-tile feature normalisation by @buglakova2025tiling. There the response is an empirical swap of one normalisation for another rather than a statement of *when* the tiled result equals the untiled one. We have not run any learned pipeline, so we claim only the shape of the argument: a correctness condition on the parameter chain, a cheap test for it, and a metric that reports whether a published number can be reproduced on someone else's harness. The design rule in items 1–4 is domain-neutral; the evidence in this paper is not.

# 9 Conclusion

We set out a correctness condition for tiled evaluation: a parameter derived from the whole image cannot be used as if it were local. We named the resulting failure mode, separated it from the benign border artefact that overlap haloes do fix, proved when each case applies, and measured both. The condition does not depend on how often the violating pattern is deployed, and Section 6.10 reports that we did not find it on the main support-selection path of the public implementations we could audit. At $n=10$ the two global-quantile reference implementations leave 6.5 % and 20.5 % of pixels differing by more than one 8-bit level at a 32-pixel halo, and the difference sits in tile *interiors*. At $N=144$ images spanning museum records, bundled reference photographs and deterministic phantoms the same plateau holds: on the 112 museum records the medians are 6.19 % and 12.77 % at halo 32, and neither implementation is exactly reproducible on more than 6 of those records.

The operator we introduce replaces the global quantile with a bounded-support local reference. It is exact on 110/112 museum records under the float-domain criterion of Section 4.1 (141/144 overall, the three exceptions being one pixel each at 1.2–3.3 levels), and it pays almost nothing for that: significantly better on SSIM and coherence/gradient retention, significantly worse on one flatness metric, indistinguishable on the rest. It also gains something not otherwise claimable. Its published quality numbers move by zero to the four decimals reported (largest residual $1.2\times10^{-4}$) when the evaluation harness changes, whereas a global-statistic variant's SSIM comparison against its sibling flips sign under the same change.

Three further results close off attractive-looking routes. The halo ladder is not monotone for a class-B statistic (27/144 images worsen when overlap doubles), so overlap cannot be tuned towards correctness. The sharp reading of the locality proposition is false for real implementations: the sufficient halo must be measured, because $2r+\rho$ understates the footprint of an operator that still contains unbounded Gaussian smoothing. And the decision statistic is not low-rank. The $2\times2$ second moment is rank-bounded by the domain dimension and its rank-1 truncation error is exactly $\lambda_{\min}$, which exceeds half the spectrum on 17–71 % of pixels and vanishes on none, so no factorisation buys the property.

What remains is a design rule, an instrument that makes violations visible at the quantisation floor, an operator that satisfies the rule, and a cost bound saying that tiling buys memory and never time: $W_{\mathrm{tiled}}\le(1+2h/T)^{2}W_{\mathrm{whole}}+Nc_0$, 1.36–2.23× in time for 4.9–8.9× less peak memory. At $4096^{2}$ (CA-AKF, the only method run at that size) and at native $4000\times3853$ (every method we ran there) the whole-canvas evaluation that this literature treats as ground truth is not slow but unavailable on an 8 GB host. None of that is a new way to tune a filter.

# Declarations

**Funding.** The author received no specific funding, grant or other financial support for the research, authorship and publication of this work.

**Competing interests.** The author declares no competing interests.

**Ethics approval and consent to participate.** Not applicable: the study uses no human subjects, personal data or animal material. All stimuli are public-domain or CC0 images downloaded from the Metropolitan Museum of Art Open Access programme, from the scikit-image data bundle and synthetically generated phantoms; the per-file licence statements are in the released manifest (`abstrabench/manifest_met.csv`, `abstrabench/README.md`).

**Consent for publication.** Not applicable (no identifiable personal data or images of persons are reported).

**Author contributions.** Y. Wang conceived the study, derived the theoretical results, wrote and ran all experiment and analysis code, prepared the figures, and drafted and approved the manuscript.

**Acknowledgements.** None.

# Data and code availability

All images are licence-clean and machine-acquired; nothing was crawled from a web page or hand-labelled. The pool is **286 images**: 254 public-domain records from the Metropolitan Museum of Art Open Access programme (CC0; acquired through the official JSON API, one manifest row per object recording the exact API field that established public-domain status (`isPublicDomain=true`) plus the object identifier, landing page, department, medium, original and served dimensions, and a SHA-1 content hash; department split European Paintings 145, Photographs 44, Costume Institute 40, Islamic Art 25), 10 bundled public-domain reference images generated locally from the scikit-image data bundle, and 22 deterministic synthetic phantoms generated by the accompanying script under a fixed seed. Served files are normalised to a longest side of 1024 px; two Met originals were additionally retrieved at full native resolution ($4000\times3853$ and $3630\times4000$, CC0) for the single-image large-canvas experiment of Section 6.7.1. `abstrabench/images_native/provenance.json` records, per image, the SHA-1 of the *retrieved JPEG original* (the licence and content anchor) with the crop offset, the decoded shape, and the digests of the two PNGs the included `make_native.py` derives from those exact pixels (full-resolution, and a centred $2048\times2048$ crop); the PNGs are not separate retrievals.

Usage is deliberately not one number. The full metric battery (six quality metrics, border bands and grid-line visibility, per image and method) runs on 10 images; the tile-consistency claim proper is re-measured on 144 images (112 Met records at the Section 6.8 cutoff plus every bundled reference and every phantom); the stratified, runtime, Proposition-1 and exactness audits over those 576 records, including the named weakest cases, are in `out/broad_qa.md`.

Builder, manifest and verification: `abstrabench/scripts/build_benchmark.py`, `abstrabench/scripts/fetch_strata.py`, `abstrabench/manifest_met.csv`, `abstrabench/README.md` (licence statements per source, verbatim), and the licence audit `abstrabench/logs/license_audit_final.json`. Implementation, sweep scripts and figure generators: `src/tc.py`, `src/metrics.py`, `src/structure.py`, `run_sweep.py`, `run_broad.py`, `run_rad.py`, `run_c.py`, `run_mem.py`, `run_giga.py`, `run_native.py`, `make_native.py`, `run_cv.py`, `probe_cv_det.py`, `probe_cv_bug.py`, `analyse.py` (Figures 1–4), `make_f6.py` (Figure 5), `make_f8.py` (Figure 6), `analyse_broad.py` (Figure 7), `analyse_cv.py` (Figure 8), `analyse_giga.py`, `analyse_theory.py` (Sections 3.1–3.2 tables), `build.py`, plus the two supervisor scripts (`broad_supervise.sh`, `native_supervise.sh`). **Running the released code.** `src/tc.py` imports the anisotropic-Kuwahara operator implementation from `code/reference/`, which is included in the release under its original MIT licence; point `CAAKF_REF` at that directory before running any sweep. No implementation has to be requested from the authors. With that variable set and the pinned environment of the reproducibility notes, the filter runs behind Sections 6.1–6.8 re-execute, the released per-image records re-derive every table and figure, and the third-party benchmark of Section 6.9 re-runs end to end, because `run_cv.py` drives OpenCV 5.0.0 directly and needs no operator code. Timing and memory figures were recorded with one evaluation mode per process on a shared four-core host and are load-dependent; peak RSS, not wall time, is the quantity the argument uses. The per-image records behind every number (`out/*.jsonl`, including `out/native.jsonl`, `out/cv_all.jsonl`, `out/cv_det.jsonl` and `out/cv_bug.jsonl`, and the `out/*.json` summaries) are released unaggregated so that any table can be re-derived, including on the images where our claim is weakest; the sole exception is the Section 4.5 mechanism probe, which has no released record and for which Section 6.3 therefore states no numbers. The public repository URL and its archival DOI are supplied in the camera-ready version; this submission is accompanied by the complete release directory and its SHA-256 manifest.

# Appendix A. Provenance of the reported numbers

Every table and figure in this paper is a deterministic reduction of a released per-image record set. The records are JSONL, one line per (image, method) or per (canvas, mode) pair, and they are released unaggregated so that any number below can be recomputed, including on the images where the claim is weakest.

**Table A1.** Provenance map: every table and figure of the paper, the released record set it is computed from, and the scope of that set.

| item | released records | scope |
|--------------------------|--------------------------|--------------------------|
| Table 2, Section 3.1 | `out/theory.json` (`p3_coherence`) | 10 images, $384\times384$, box radius 8 |
| Table 3, Section 3.2 | `out/mem.jsonl`, `out/mem_manual.jsonl`, `out/giga_CA-AKF.json` | whole-canvas and tile-wise passes, one mode per process |
| Tables 4–7, Figures 1–4 | `out/sweep.jsonl`, `out/sweep_final.jsonl` | 10 images, $384\times384$, halos 0–32 px, grids $n\in\{2,4,8\}$ |
| Table 8 | `out/rad.jsonl` | 4 images, $\rho\in\{4,8,16,32\}$, 8-px halo ladder to 40 px |
| Table 9 | `out/c.jsonl` | 4 images, $c\in\{1,1.5,2,3\}$, $\rho=8$ |
| Tables 10–11, Figure 5 | `out/mem.jsonl`, `out/mem_manual.jsonl`, `out/giga_*.json` | $2048^2$ and $4096^2$ canvases, halos 16 and 48 px |
| Table 12, Figure 6 | `out/native.jsonl` | native $4000\times3853$ scan and its $2048^2$ crop |
| Tables 13–14, Figure 7 | `out/broad_all.jsonl` | 144 images, 4 methods, halos 0/16/32 px |
| Table 15, Figure 8 | `out/cv_all.jsonl`, `out/cv_det.jsonl`, `out/cv_bug.jsonl` | 144 images, seven OpenCV 5.0.0 algorithms, halos 0–48 px |
| Table 16, Section 6.10 | archived source trees (`gpuakf`, `polyakf`, `TD-Anisotropic-Kuwahara`, JLR operator base) | four implementations, one release each, parameter expressions read at quoted line ranges |
| Table 17, Section 6.11 | `out/sweep_replication.jsonl` | 23 images on the pinned Windows stack, 4 methods, 20,000 bootstrap resamples |
| Table 6 effects and intervals | `out/sweep.jsonl`, `out/sweep_final.jsonl` | 10 paired images, 20,000 bootstrap resamples, seed 20261104 |

Line ranges behind Table 16: `gpuakf` `glsl/akf_v2.glsl:6-8,90-91`, `glsl/tfm.glsl:24-25`, `glview.cpp:399-402,465-490`; `polyakf` `akf_v2n4.glsl:7-9,83-84`, `polyakf.cpp:292-319`; JLR operator base `filters.py:148`, `repair.py:135`, `methods.py:323`. The TD archive contains four `.tox` binaries and a `README.md` only.

Two auxiliary documents are released alongside the records and are cited in the text only through this appendix: the literature-search log (query strings, indexes and DOIs behind the absence-of-evidence claim in Section 6.9) and the reconciliation report for the 576 records of Section 6.8, which re-derives every aggregated cell from the raw lines and records a pass for all 4 rows × 8 cells. The licence audit for the 286 benchmark images is released with the benchmark itself, one row per object, with the API field that established public-domain status.
