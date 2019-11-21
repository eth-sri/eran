#include "domains/interval.h"
// #include "domains/polyhedra.h"
#include <string>
#include <fstream>
#include <vector>

#pragma once

using namespace std;

template <class T>
class Pixel {

public:
    T x;
    T y;
    int channel;

    Pixel(T x, T y, int channel) : x(x), y(y), channel(channel) {}
};

class Image {
  
public:
  Interval a[35][35][3];
  int nRows, nCols, nChannels;
  double noise;

  Image(int nRows, int nCols, int nChannels) {
      this->nRows = nRows;
      this->nCols = nCols;
      this->nChannels = nChannels;
      this->noise = 0;
  }
  Image(int nRows, int nCols, int nChannels, std::string line, double noise);
  void print_ascii() const;
  vector<double> to_vector() const;
  void print_csv(std::ofstream& fou) const;

  Interval find_pixel(int x, int y, int i) const;
  Pixel<double> getPixel(double r, double c, int i) const;
};


class Statistics {

public:
    Pixel<int> active_px = {0,0,0};
    std::vector<std::vector<std::vector<int>>> counter;
    double tot_poly_dist;
    int num_poly;

    Statistics() {
        this->active_px = Pixel<int>(0,0,0);
        this->counter = std::vector<std::vector<std::vector<int>>>(32, std::vector<std::vector<int>>(32, std::vector<int>(3, 0)));
        this->tot_poly_dist = 0;
        this->num_poly = 0;
    }

    double getAveragePolyhedra() {
        return tot_poly_dist / num_poly;
    }

    void updateAveragePolyhedra(double mean) {
        tot_poly_dist += mean;
        num_poly += 1;
    }

    void inc();

    int total_counts() const;

    void zero();
};
