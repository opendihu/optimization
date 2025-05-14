# Isotonic contraction models

## Description

### Optimize prestretch force
In this case we add neumann boundary conditions that simulate the muscle being stretched with a constant force before the contraction process. 

### Optimize prestretch length
In this case we simulate the muscle being stretched by a given distant by displacing all the nodes equidistantly.

### Optimize stress free length
In this case use the prestretch-length simulation to stretch muscles to a certain length and compare the muscles' properties for different starting lengths.

## Differences
In the third case we compare different starting lengths, whereas in the first two we compare different lengths after prestretch but with the same starting length. 

The first two cases have the same idea with different implementations. The main difference is the stability of the prestretch simulations. With the prestretch-force simulation you can stretch the muscle until it tears, so it is stable for all realistic prestretches. The prestretch-length case is very unstable for larger prestretches and produces unrealistic results. For smaller prestretches, this simulation is as stable as the prestretch-force case. The stability limit is a prestretch-length of 15.3% of the muscle's starting length. A question is, if the muscle behaves differently after the two different prestretches. An experiment answers this question: We simulate a few stretching and contraction processes with the prestretch-force simulation, and do the same with the prestretch-length simulation, where we use the same prestretch lengths we got from the prestretch-force simulation. We then compare the contraction lengths. As an example: For 5N prestretch force we get a prestretch length of 0.68321822237417cm and a contraction length of 0.8592888447139337cm. If we do the prestretch-length simulation with 0.68321822237417cm, we get a contraction length of 0.8592888447167635cm. This is a difference of 3e-11. On average, the difference in contraction length is about 10⁻¹⁰cm. Therefore, we can anwer the question if the muscles behave differently after the different prestretch kinds as a clear no. This also means, that we can use the stable prestretch simulation exclusively. 