#include "INNC/utils/rand.hpp"

namespace INNC{
std::mt19937 rng{std::random_device{}()};
}
