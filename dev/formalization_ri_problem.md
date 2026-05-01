## Formalization of the Reject inference problem

The RI problem is a specific case of missing data and it gets handled
as such in the literature. However as other authors have noted, a robust
assesment of the validity of the RI-methodology and its limits requires
a good definition of the kind of missigness assumed in the data. Although
many examples exist on some approaches how to formalize the missigness,
like Ehrhardt et. al (2020), they often fall short in the measure-theoretical
formality. This section aims to propose a new way to describe the RL-problem
closing this gap, as even if it looks convoluted it helps to be more
transparent on the assumptions of the DGP and to more easily describe
the used simulation framework. Furthermore such level of formality might
result helpful in future work to derive mathematical results from it, as
stochastics from a measure theoretic perspective is the literature standard
for new (influential) results.

Before starting with the mathematical notation, it is helpful to describe in
words how the missigness mechanism occurs, to be very precise on the meaning
of each concept which will be later translated as random variables, their realizations and measurable functions.

Considering the heavy focus on measure theory of this section, it makes sense
to start by characterizing the random variables through which the credit data
generating process could be described. To avoid the need of further notation
I assume that all information that all information of an applicant that has any
dependence with its future repayment status and the accepta can be represented as a real number.


Let $X_o, X_h, Y, Y_o$ and $Z$ be random variables on the probability space $(\Omega, \mathcal{F}, \mathbb{P})$. Define 
$$
X_i : (\Omega, \mathcal{F}) \to \left(\mathbb{R}^{f_i}, \mathcal{B}(\mathbb{R}^{f_i}) \right), f_i \in \mathbb{N}, i \in \{o, h\}
$$
be the random variables from which the observed (i. e. information that was asked and saved for score-card estimation in the data-base of a financial institution) and hidden (i. e. information not used for score-card estimation) features from an applicant come from. The joint distribution from the full features vector is characterized by
$$
X : (\Omega, \mathcal{F}) \to \left(\mathbb{R}^{f}, \mathcal{B}\left(\mathbb{R}^{f} \right) \right), f:=f_o + f_h
$$
with
$$
\forall \omega \in \Omega: X(\omega) := \begin{pmatrix} X_o(\omega) \\ X_h(\omega) \end{pmatrix}
$$
Further with $V$ a placeholder for $Y, Z$ and $Y_o$ let
$$
V : (\Omega, \mathcal{F}) \to \left( E_V, 2^{E_V} \right)
$$
with
$$
E_Y = \{g,b\}, \quad , E_Z = \{a,r\} \quad E_{Y_o} = \{g,b, \texttt{na}\}
$$

be the random variables describing the repayment type, acceptance status and observable repayment status of the loan defined $\forall \omega \in \Omega$ as
$$
Y(\omega) = \begin{cases}
    b & \text{if applicant is a defaulter} \\
    g & \text{if applicant is a payer}
\end{cases} ,
$$
$$
Z(\omega) = \begin{cases}
    a & \text{if applicant gets accepted (financed)} \\
    r & \text{if applicant gets rejected (non-financed)}
\end{cases}
$$
and
$$
Y_o(\omega) = \begin{cases}
    Y(\omega) & Z(\omega) = a \\
    \texttt{na} & Z(\omega) = r
\end{cases}
$$

$Z$ is of course a random variable only in the extent that is dependent on $X_o$ and $X_f$. To formalize this we can define two further maps, namely
$$
s : \mathbb{R}^{f_o} \to \mathbb{R}, \; 
\mathcal{B}\left(\mathbb{R}^{f_i}\right)-\mathcal{B}\left(\mathbb{R}\right) \text{ measurable}
$$
as the scorecard and
$$
d : \mathbb{R} \times \mathbb{R}^{f_h} \to \{a, r\}, \;
    \left(\mathcal{B}\left(\mathbb{R}\right)
    \otimes
    \mathcal{B}\left(\mathbb{R}^{f_i}\right)\right) - 2^{\{a, r\}}
    \text{ measurable}
$$

as the (acceptance) decision function. Then we can further define $Z$ as

$$
\forall \omega \in \Omega:
Z(\omega) =
d\left(
    s\left(X_o(\omega)\right),
    X_h(\omega)
\right)
$$
