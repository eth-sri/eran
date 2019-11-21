#include "dsl.h"

//Point::Point(Interval x, Interval y) {
//  this->x = x;
//  this->y = y;
//}
//
//Image SpatialTransformation::operator()(const Image& img, BilinearInterpolation interpolation) {
//
//}
//
//Translation::Translation(double dx, double dy) {
//  this->dx = dx;
//  this->dy = dy;
//}
//
//Point Translation::operator()(Point p) {
//  return {p.x + this->dx, p.y + this->dy};
//}
//
//Point Translation::backward(Point p) {
//  return {p.x - this->dx, p.y - this->dy};
//}
//
//std::ostream& operator<<(std::ostream& os, const Point& p) {
//  return os << p.x << " " << p.y;
//}
//
//std::ostream& operator<<(std::ostream& os, const Translation& t) {
//  return os << "Translation(" << t.dx << "," << t.dy << ")";
//}
//
