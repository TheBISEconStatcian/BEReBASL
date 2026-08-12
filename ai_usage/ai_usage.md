# Transparency on AI usage

For the master's thesis the following AI's have been used

* ChatGPT (Version 5 - free usage)
* Copilot by Bing powered with ChatGPT5.
* Perplexity
* Claude
* GitHub Copilot
* DeepAI

For each used chat a link is provided to be called if wished. I always add a small description on the general topic of the chat. When I see it necessary/sensible I also add some more notes on the purpose/themes of the chat as sub-bullet points. Further small chats have been used regarding variable
naming, small refactoring, synonym finding, reformulation of single sentences or asking small and
general statistics/stochastics/ML/programming questions which have not been documented as I
do not see them as neither important nor anywhere near critical contributions of AI in the
master's thesis project. Thereby I mean that AI was used for purposes where a couple of year's
ago I would've used a dictionary, searched my lecture notes, read in a book, looked in the
official documentation, serached in some forum like stack overflow or just asked a friend for his
opinion on naming.

## Links to `ChatGPT` usage

- [Understanding and implementation of GaussianMixture sampling](https://chatgpt.com/share/69411a52-58d0-8008-9a1e-fc2d0abce3a6)
- [`torch.device` nuances and help on docstring documentation.](https://chatgpt.com/share/69411b62-403c-8008-ad16-5a7a6d3890d0)
- [Implementation of batched ROC-AUC to match torch.metrics.BinaryAUROC behavior](https://chatgpt.com/share/696a9ebb-da1c-8008-b3bc-c0a2a0b621ea)
- On the proof for the train-splitting over valid entries and normalized shapes
    - [Chat with the full problem](https://chatgpt.com/share/69872f18-124c-8008-b645-dc2cf54764f5)
    - [To refine a step that was wrong in the first proof.](https://chatgpt.com/share/69874c4c-c3f8-8008-967b-5b31b597956b)
- [Help on defining RI's set up random variables](https://chatgpt.com/share/69f5ba4b-4324-83eb-ba3b-8cf1575aca87)
- [Start developing what would end up being the pairwise plotting of CreditDataGenerator](https://chatgpt.com/share/69f5c1bb-9838-83eb-9ec2-2d82c088bab3)
- [Questions on Coverage probability of MVN](https://chatgpt.com/share/69f5c191-2498-83eb-94e0-71de16de3448)
- [Stochastically clean understanding of Gaussian Mixtures](https://chatgpt.com/share/69f5c234-3f70-83eb-9caf-2663da7039b9)
- [Adapting the CV-Based acceptance loop script into a CLI ready script following the original base acceptance loop](https://chatgpt.com/share/69f5c2c3-30fc-83eb-8420-c413d840d6eb)
- [Weight the option of tensorizing AU-ROC calculation](https://chatgpt.com/share/69f5c31f-8198-83eb-8ce5-ee4925f22b88)
- [Help with index gymnastics in batched au-roc development](https://chatgpt.com/share/69f5c364-7318-83eb-8813-2239f056b40f)
- [Redaction of the description in words and a stochastic model of the RI problem](https://chatgpt.com/share/69f9b0c6-b730-8326-819d-40c9c124184c).
  Considering that using AI for writting can come close to plagiarism, a very close commit-wise book-keeping is kept here
  to show the *exact* extent of the usage
  - [49dbafd514d89c967b0e0040fbcf1ce826b137b8](https://github.com/TheBISEconStatcian/BEReBASL/commit/49dbafd514d89c967b0e0040fbcf1ce826b137b8):
    Corrections and refinements answering specific questions to parts of my text. It also
    includes insights from a question in a [perplexity chat](https://www.perplexity.ai/search/261ac5f8-71b6-4520-b949-ca2e1ea6b741)
    (also reported in the perplexity subsection).
  - [66f2f330b35c77dcdaa022141fdff66f6d2ef472](https://github.com/TheBISEconStatcian/BEReBASL/commit/66f2f330b35c77dcdaa022141fdff66f6d2ef472):
    Copy-pasting the corrections of ChatGPT based on my complete input of the subsection "Concepts definitions" of section
    "Theoretical background and problem setting".
  - [0781b5b894d16f561c34a9e93e78b65a8d8fcff7](https://github.com/TheBISEconStatcian/BEReBASL/commit/0781b5b894d16f561c34a9e93e78b65a8d8fcff7):
    Refinements to the recommendations of ChatGPT done by the me.
  - [b6e96d383b646e8e34116c72f20560ddc17ced97](https://github.com/TheBISEconStatcian/BEReBASL/commit/b6e96d383b646e8e34116c72f20560ddc17ced97):
    Help on lowering redundancy and fluency of the dynamic description of the acceptance process.
  - [9b87490cafc907cac86fea25a4dcd69918f9e553](https://github.com/TheBISEconStatcian/BEReBASL/commit/9b87490cafc907cac86fea25a4dcd69918f9e553):
    Refinements to the corrections of the last bullet point.
  - [2d6c568915c0cbc270d41ccf655eab01846d50f4](https://github.com/TheBISEconStatcian/BEReBASL/commit/2d6c568915c0cbc270d41ccf655eab01846d50f4):
    Corrections of the measure theoretic description, with which I agreed quickly.
  - [620c571e853e00fecda86d541e0f41c7ce183571](https://github.com/TheBISEconStatcian/BEReBASL/commit/620c571e853e00fecda86d541e0f41c7ce183571) corrections on the reformulation of stochastic process.
  - [3b3efa4a239be16c7236a7bbbc0d09c9be49bc69](https://github.com/TheBISEconStatcian/BEReBASL/commit/3b3efa4a239be16c7236a7bbbc0d09c9be49bc69) Corrections on nuances of the family definitions, regularity conditions argument was inspired by ChatGPT and checked against with the given source.
  - [0820738157fabcd501bb189b7457c47535bebaf5](https://github.com/TheBISEconStatcian/BEReBASL/commit/0820738157fabcd501bb189b7457c47535bebaf5) Polishing the learn algorithm measurability argument. (Apply suggestions)
  - [357b737544a38a828e5e3ca42371004f646b1b9b](https://github.com/TheBISEconStatcian/BEReBASL/commit/357b737544a38a828e5e3ca42371004f646b1b9b) Apply redaction and precision remarks as discussed with
  the chat.
- [Metaprompting the MNAR adaptation for Claude](https://chatgpt.com/share/6a0165fb-7310-83eb-a67d-ef86851dd78d)
- [Implementation of `PerfectBayesClassifier`](https://chatgpt.com/share/6a133ae4-7ee4-83eb-a3aa-d301b268f725)
- [Shortening and sharpening the unsurprising setting subsusbsection](https://chatgpt.com/share/6a27e009-b740-83ed-b931-37ea0b8c6929). The copied text from the suggestions of the LLM can be seen in
  - [7891f07de6887da0b78913720835801ab1154492](https://github.com/TheBISEconStatcian/BEReBASL/commit/7891f07de6887da0b78913720835801ab1154492)
  - [f654c656c9428e493f81e44774bac9a06fc56175](https://github.com/TheBISEconStatcian/BEReBASL/commit/f654c656c9428e493f81e44774bac9a06fc56175)
  - [9b8eb62526a026cd86d149778d781a1d583e3da8](https://github.com/TheBISEconStatcian/BEReBASL/commit/9b8eb62526a026cd86d149778d781a1d583e3da8)
- [Tackling down vectorization for inserting piecewise 0 regions for region wise integration, commenting and readability of function](https://chatgpt.com/share/6a350308-bf90-83eb-80ea-34350ebb07fa). Commits
  - [36b60f2bfb09e11864a40a8c0682c5c55246d1b8](https://github.com/TheBISEconStatcian/BEReBASL/commit/36b60f2bfb09e11864a40a8c0682c5c55246d1b8)
  - [5a621a767d71690946400653c56d4117f0ee67dd](https://github.com/TheBISEconStatcian/BEReBASL/commit/5a621a767d71690946400653c56d4117f0ee67dd)
  - [a201b10ed9ae47595d9da4dec27c90d7496a0e8b](https://github.com/TheBISEconStatcian/BEReBASL/commit/a201b10ed9ae47595d9da4dec27c90d7496a0e8b)
- [Rewriting the (first) abstract](https://chatgpt.com/share/6a390caf-0204-83eb-bfd7-7fbe861fef2b)
- [Solving positive definiteness space](https://chatgpt.com/share/6a3cfe56-1798-83eb-b376-a052dd725154)
- [Rewriting again the abstract](https://chatgpt.com/share/6a7aec6d-cbd4-83eb-bfc0-d6acecef53f2)
  - Original proposal: [64c7f01ca3cecafda818b8ab3639996f9a1fa34f](https://github.com/TheBISEconStatcian/BEReBASL/commit/64c7f01ca3cecafda818b8ab3639996f9a1fa34f)
  - Corrections with AI generation: [f3a8e97122e7a1beafb89be59f210ba0a83b8e1a](https://github.com/TheBISEconStatcian/BEReBASL/commit/f3a8e97122e7a1beafb89be59f210ba0a83b8e1a)


## Links to `Copilot` usage

- [Technicalities about sampling with `torch.Generator`](https://copilot.microsoft.com/shares/8aPAPMiRK6KnCoTRMDr34)
- [Nuances between the difference of `torch.Tensor.expand` vs `torch.full` for singleton tensors](https://copilot.microsoft.com/shares/QwmVEjefZbmFM2ULYPB11)
- [Tensor and pointer arithmetic within `pytorch` for development of `GaussianMixture`](https://copilot.microsoft.com/shares/QwmVEjefZbmFM2ULYPB11)
- [Help on designing and documenting the CreditData class](https://copilot.microsoft.com/shares/wbyEnAbf9rrAhBcMcWPnY)
- [Understanding better the unbiased bad ratio calculation](https://copilot.microsoft.com/shares/dbwim2EaiZE4UxRxkuURcunt)
- [Help with `CreditData` docstrings, iris-data-set retrieval and implementation of Logit in torch](https://copilot.microsoft.com/shares/6ZQt1YN531MpHUZjHipFk)
    - The purpose was to ensure a torch based option to match R's GLM behaviour
- [How to set up repo to look profesional and be an importable package](https://copilot.microsoft.com/shares/oWZVD4bAURP55UEFk2p92)
- [Expanding labeling reject of basl for n-dimensioned batch shapes](https://copilot.microsoft.com/shares/52rQiyM8NVkCNWAg59AGV)
- [Vectorized way for "compact" appending and resizing after labeling in `CreditDataSample`](https://copilot.microsoft.com/shares/V8TcEqDX58X18DBmRiunc)
- [Documentation updating after batch handling and custom nan value](https://copilot.microsoft.com/shares/V8TcEqDX58X18DBmRiunc)
- [Vectorized labeling for BASL](https://copilot.microsoft.com/shares/V8TcEqDX58X18DBmRiunc)
- [Generalize train test splitting to new batched logic](https://copilot.microsoft.com/shares/oipQ7Yv7ddGRUyULHtVZA)
- [List available torch devices and docstring of `CreditDataSample.inspect_data`](https://copilot.microsoft.com/shares/JZD2AY81aEyaaSYnpstyq)
- [Bettering my `CreditDataSample.filter_unlabeld` method](https://copilot.microsoft.com/shares/PCBB8Q7nqNwHwuw2rdEJy)
- [Parser for the acceptance loop](https://copilot.microsoft.com/shares/2bCrkT5X8Ziii7JZE8PNr)
- [Augmenting the batched au-roc and masked trapz to support arbitrary dim + docstrings](https://copilot.microsoft.com/shares/B96zzDAVgaDsGSFPeF8JZ)

## Link to `perplexity` usage

- [Designing logic for handling "batched CreditDataSamples" (labels.dim() > 1)](https://www.perplexity.ai/search/in-this-function-i-want-to-kee-hISfZ3XFQ6KMTW1Kyy6h2w#2)
- [Small literature research and help understanding MAR vs MNAR](https://www.perplexity.ai/search/261ac5f8-71b6-4520-b949-ca2e1ea6b741)
- [Trying to find literature on ML consistent estimators under MAR proof and trying to bring formalism for hypothesis fromulation](https://www.perplexity.ai/search/in-the-credit-risk-literature-OxEvKj8cRJmjW8C0WABg2g)
- [Relationship between MAR and the conditional density of $Y|X$ given the acceptance status](https://www.perplexity.ai/search/well-i-am-realizing-that-from-lUQoNVX3Rt.ZnaDOz6gzLg)

## Links to `Claude` usage

- [Development of confusion probability measure](https://claude.ai/share/a9be6937-d7bb-4574-83bd-5a6f36e4b228) Also includes related topics like
    - Precise definition of the measurement
    - Understanding of the bayesian error rate as a related measure
    - Calculation and implementation of log probs of gaussian mixture
    - Production of bayes_classifier_math.md
- [Development of `BatchedLogistic` class](https://claude.ai/share/cc21b148-7308-4b77-ab23-33fe14c38c7c)
- [Development of `k_fold_cv_normalized_split_batched`](https://claude.ai/share/862fe75b-b334-48ce-b489-5c2e58b5b7cc)
- [Finding and implementing metric for CV-based threshold](https://claude.ai/share/27b633c3-c6a2-4cba-b540-9158037fdf0d)
- [Update CV-Acceptance loop CLI as ChatGPT was unable](https://claude.ai/share/9e4ef49a-6346-46f4-9237-f9080d0c845d)
- [Include MNAR logic according to my own code exemplifying the wished logic](https://claude.ai/share/7c92b2bc-da35-45f9-a216-ab1ed9d4237d)

## `Github-Copilot`

Most of the help used with Github copilot was done solely with the chat function.
However, while developing in some cases the auto-fill function was activated, which
constitutes AI usage. The latest has not been documented. The chats have been saved
under `./gihub_copilot_chats` and have self explaining names. Only for chats which
seem to need a small clarification a small description is added here:

* `help_developing_simulation_diagnostics_functions.md`: With diagnostics functions
  are meant the graphics shown in `~/path_to_repo/dev/hypothesis_development.ipynb`
  depicting the comparision of expectation vs realized performance and the diagnos-
  tics on accepts and rejects (rates or counts) + the respective functions saving
  the grid of all experiments.

One chat was lost which helped developing further the 3D representation of the
CreditDataGenerator DGP, it contained refinements to the worked done by the saved
chat.

## Links to `Gemini`

* [Decide which rendering tool to use and set up quarto project](https://gemini.google.com/share/c73006f0efb8)

## Links to `Math AI` by [`DeepAI`](https://deepai.org/)

* [Help organizing the idea of convergence of the accepts population under MAR assumptions.](https://deepai.org/chat/mathematics#f1ec7e34-48ec-4059-a123-a9134a7dd1c2) The commits corresponding to the respective usage of the generated content are
  * [d879a3956d35d2217306c88fa655115bf53056dc](https://github.com/TheBISEconStatcian/BEReBASL/commit/d879a3956d35d2217306c88fa655115bf53056dc)
  * [85b48080ca356cc855f33af1513503a675c98d1a](https://github.com/TheBISEconStatcian/BEReBASL/commit/85b48080ca356cc855f33af1513503a675c98d1a)
  * [a50f1acdac1acf7bc2292b94cae7ef685f829178](https://github.com/TheBISEconStatcian/BEReBASL/commit/a50f1acdac1acf7bc2292b94cae7ef685f829178)