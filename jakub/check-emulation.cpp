#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

using short_t = int8_t;
using long_t = int16_t;
using ulong_t = uint16_t;

extern "C" long_t emulate_op(long_t a, long_t b, long_t emulate);

void fatal(const std::string &msg) {
  fprintf(stderr, "Fatal error: %.*s\n", static_cast<int>(msg.size()),
          msg.c_str());
  fflush(stderr);
  abort();
}

int64_t parseNum(std::string_view num) {
  if (num == "i8_min")
    return std::numeric_limits<int8_t>::min();
  if (num == "i8_max")
    return std::numeric_limits<int8_t>::max();
  if (num == "u8_min")
    return 0;
  if (num == "u8_max")
    return std::numeric_limits<uint8_t>::max();
  if (num == "i16_min")
    return std::numeric_limits<int16_t>::min();
  if (num == "i16_max")
    return std::numeric_limits<int16_t>::max();
  if (num == "u16_min")
    return 0;
  if (num == "u16_max")
    return std::numeric_limits<uint16_t>::max();

  std::string buf(num);
  std::istringstream ss(buf);
  int64_t res = 0;
  if (ss >> res) {
    if (res > int64_t(std::numeric_limits<ulong_t>::max()))
      fatal("Number too large: " + std::to_string(res));
    if (res < int64_t(std::numeric_limits<long_t>::min()))
      fatal("Number too large: " + std::to_string(res));

    return res;
  }

  fatal("Failed to parse '" + std::string(num) + "'");
  return 0;
}

std::pair<int64_t, int64_t> parseInterval(std::string_view interval) {
  const size_t dash = interval.find('-');
  if (dash == std::string_view::npos)
    fatal("Could not parse interval: '" + std::string(interval) + "'");

  const std::string_view first = interval.substr(0, dash);
  const std::string_view second = interval.substr(dash + 1);
  const int64_t low = parseNum(first);
  const int64_t high = parseNum(second);
  if (high < low)
    fatal("Bad interval: [" + std::to_string(low) + ", " +
          std::to_string(high) + "]");

  return {low, high};
}

int main(int argc, char **argv) {
  if (argc < 3)
    fatal("Not enough args\n");

  auto [outerLow, outerHigh] = parseInterval(argv[1]);
  auto [innerLow, innerHigh] = parseInterval(argv[2]);
  const int64_t numOuter = outerHigh - outerLow + 1;
  const int64_t numInner = innerHigh - innerLow + 1;

  printf("Space: [%ld, %ld] x [%ld, %ld] ==> %ld checks\n", int64_t(outerLow),
         int64_t(outerHigh), int64_t(innerLow), int64_t(innerHigh),
         numOuter * numInner);

  unsigned lastProg = -1;
  for (int64_t i = 0; i < numOuter; ++i) {
    unsigned progress = 100 * i / numOuter + 1;
    if (progress != lastProg) {
      auto bar = std::string(progress, '=') + std::string(100 - progress, ' ');
      printf("\r[%.*s] %u%% complete", static_cast<int>(bar.size()),
             bar.c_str(), progress);
      fflush(stdout);
      lastProg = progress;
    }

    for (int64_t j = 0; j < numInner; ++j) {
      const int64_t x = i + outerLow;
      const int64_t y = j + innerLow;
      const long_t argA(x);
      const long_t argB(y);
      const long_t res_wide = emulate_op(argA, argB, 0);
      const long_t res_emulated = emulate_op(argA, argB, 1);
      if (res_wide == res_emulated)
        continue;

      printf("\n=========== Op %ld, %ld\n", x, y);
      printf("wide:     %ld\n", int64_t(res_wide));
      printf("emulated: %ld\n\n", int64_t(res_emulated));
    }
  }

  printf("\nDone\n");
  return EXIT_SUCCESS;
}
