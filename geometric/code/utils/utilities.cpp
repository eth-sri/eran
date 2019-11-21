#include "utilities.h"
#include <cassert>
#include <cmath>
#include <iostream>

Image::Image(int nRows, int nCols, int nChannels, std::string line, double noise) {
    this->nRows = nRows;
    this->nCols = nCols;
    this->nChannels = nChannels;
    this->noise = noise;
    string curr;
    int j = 0;
    bool first = true;

    vector<Interval> its;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == ',') {
            if (!first) { // first denotes label
                its.push_back(Interval(stod(curr) - noise, stod(curr) + noise).meet(Interval(0, 1)));
            }
            first = false;
            curr = "";
            ++j;
        } else {
            curr += line[i];
        }
    }
    its.push_back(Interval(stod(curr) - noise, stod(curr) + noise).meet(Interval(0, 1)));

    assert((int)its.size() == nRows * nCols * nChannels);
    int nxt = 0;
    for (size_t r = 0; r < nRows; ++r) {
        for (size_t c = 0; c < nCols; ++c) {
            for (size_t i = 0; i < nChannels; ++i) {
                this->a[r][c][i] = its[nxt++];
            }
        }
    }
}

void Image::print_csv(std::ofstream& fou) const {
  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
        for (int k = 0; k < nChannels; ++k) {
            if (i != 0 || j != 0 || k != 0) {
                fou << ",";
            }
            fou << a[i][j][k].inf << "," << a[i][j][k].sup;
        }
    }
  }
  fou << std::endl;
}

vector<double> Image::to_vector() const {
  vector<double> result;
  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
        for (int k = 0; k < nChannels; ++k) {
            result.push_back(a[i][j][k].inf);
            result.push_back(a[i][j][k].sup);
        }
    }
  }
  return result;
}

void Image::print_ascii() const {
  std::cout << "====================================================================" << std::endl;
  double sum_widths = 0;
  for (int i = 0; i < nRows; ++i) {
    for (int j = 0; j < nCols; ++j) {
      std::cout << (a[i][j][0].inf >= 0.5 ? "#" : " ");
    }
    std::cout << " | ";
    for (int j = 0; j < nCols; ++j) {
      std::cout << (a[i][j][0].sup >= 0.5 ? "#" : " ");
      for (int k = 0; k < nChannels; ++k) {
          sum_widths += a[i][j][k].sup - a[i][j][k].inf;
      }
    }
    std::cout << " | " << std::endl;
  }
  std::cout << "Average width: " << sum_widths / (nRows * nCols * nChannels) << std::endl;
  std::cout << "====================================================================" << std::endl;
}

Interval Image::find_pixel(int x, int y, int i) const {
  if (x % 2 == nRows % 2 || y % 2 == nCols % 2) {
    return {-noise, noise};
  }
  int c = (x + (nCols - 1)) / 2;
  int r = (y + (nRows - 1)) / 2;
  if (r < 0 || r >= nRows || c < 0 || c >= nCols) {
    return {-noise, noise};
  }
  return a[r][c][i];
}

Pixel<double> Image::getPixel(double r, double c, int channel) const {
    return {2 * c - (nCols - 1), 2 * r - (nRows - 1), channel};
}

void Statistics::inc() {
    ++counter.at(active_px.x).at(active_px.y).at(active_px.channel);
}

int Statistics::total_counts() const {
    int ret = 0;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            for (int k = 0; k < 3; ++k) {
                ret += counter[i][j][k];
    }   }   }
    return ret;
}

void Statistics::zero() {
    int all_zeros[32][32][3];
    this->counter = std::vector<std::vector<std::vector<int>>>(32, std::vector<std::vector<int>>(32, std::vector<int>(3, 0)));
}
