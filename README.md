# Least-Effort

(C) Copyright R M L Evans 2025

An elementary implementation of the methods described in the article:
  A principle of "least effort" to describe the natural movements of animals
  R M L Evans
  Journal of Physics A: Mathematical and Theoretical (2025)
  http://doi.org/10.1088/1751-8121/ae126e

To model an animal with articulated joints, the user supplies the energy-related functions (such as power) defined in the article. The algorithm implements the variational principle described in the article by applying a standard library function for constrained minimization, in order to determine the animal's movement.

The variational principle generates graceful movements between user-defined beginning and end states without relying on data sets or human intervention to find in-between postures. Importantly, the movements respect the laws of Newtonian dynamics, while also capturing the lazy appearance of the voluntary movements of wild animals that have evolved to move efficiently.
