
### The current parallel implementation
May or may not be faster, though it's a lot more consistent on my machine.

CDA Server test: ❌
### Changes:
Removed `reduction` clause from first loop used local variable instead and took advantage of `omp critical` below. Removes reduction overhead and should make minimal impact inside critical.