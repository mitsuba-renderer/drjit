# Syntax:  cmake -P auto-stub.cmake <input> <output>
# Replaces all occurrences of 'drjit.llvm' with 'drjit.auto'

file(READ ${CMAKE_ARGV3} FILE_CONTENTS)
string(REPLACE "drjit.llvm" "drjit.auto" FILE_CONTENTS "${FILE_CONTENTS}")
file(WRITE "${CMAKE_ARGV4}" "${FILE_CONTENTS}")
