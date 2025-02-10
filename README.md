# Least-Effort

(C) Copyright R M L Evans 2025

A simple implementation of the methods described in the article 'A principle of "Least Effort" to describe the natural movements of animals'.

To model an animal with articulated joints, the user supplies the energy-related functions (such as power) defined in the article. The algorithm implements the variational principle described in the article by applying a standard library function for constrained minimization, in order to determine the animal's movement.

The variational principle generates graceful movements between user-defined beginning and end states without relying on data sets or human intervention to find in-between postures. Importantly, the movements respect the laws of Newtonian dynamics, while also capturing the lazy appearance of the voluntary movements of wild animals that have evolved to move efficiently.
