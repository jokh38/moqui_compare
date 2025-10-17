#ifndef MQI_IO_HPP
#define MQI_IO_HPP

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iomanip> // std::setprecision
#include <iostream>
#include <limits>   // std::numeric_limits
#include <numeric> //accumulate
#include <valarray>
#include <zlib.h>

#include <sys/mman.h> //for io

#include "mqi_common.hpp"
#include "mqi_hash_table.hpp"
#include "mqi_roi.hpp"
#include "mqi_scorer.hpp"
#include "mqi_sparse_io.hpp"
#include "mqi_dicom.hpp"

// DCMTK headers - only include if DCMTK is available
#if DCMTK_FOUND
  #include <dcmtk/dcmdata/dctk.h>
  #include <dcmtk/dcmdata/dcfilefo.h>
  #include <dcmtk/dcmdata/dcdeftag.h>
  #include <dcmtk/dcmdata/dcdatset.h>
  #include <dcmtk/ofstd/ofcond.h>
  #include <dcmtk/ofstd/ofstring.h>
#endif


namespace mqi {

namespace io {
///<  save scorer data to a file in binary format
///<  scr: scorer pointer
///<  scale: data will be multiplied by
///<  dir  : directory path. file name will be dir + scr->name + ".bin"
///<  reshape: roi is used in scorer, original size will be defined.
template <typename R>
void save_to_bin(const mqi::scorer<R> *src, const R scale,
                 const std::string &filepath, const std::string &filename);

template <typename R>
void save_to_bin(const R *src, const R scale, const std::string &filepath,
                 const std::string &filename, const uint32_t length);

template <typename R>
void save_to_npz(const mqi::scorer<R> *src, const R scale,
                 const std::string &filepath, const std::string &filename,
                 mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots);

template <typename R>
void save_to_npz2(const mqi::scorer<R> *src, const R scale,
                  const std::string &filepath, const std::string &filename,
                  mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots);

template <typename R>
void save_to_npz(const mqi::scorer<R> *src, const R scale,
                 const std::string &filepath, const std::string &filename,
                 mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots, R *time_scale,
                 R threshold);

template <typename R>
void save_to_bin(const mqi::key_value *src, const R scale,
                 uint32_t max_capacity, const std::string &filepath,
                 const std::string &filename);

template <typename R>
void save_to_mhd(const mqi::node_t<R> *children, const double *src,
                 const R scale, const std::string &filepath,
                 const std::string &filename, const uint32_t length);

template <typename R>
void save_to_mha(const mqi::node_t<R> *children, const double *src,
                 const R scale, const std::string &filepath,
                 const std::string &filename, const uint32_t length);

template <typename R>
void save_to_dcm(const mqi::node_t<R> *child, const double *src, const R scale,
                 const std::string &filepath, const std::string &filename,
                 const uint32_t length, const mqi::dicom_t &dcm_info,
                 bool two_cm_mode);

} // namespace io
} // namespace mqi

///< Function to write key values into file
///< src: array and this array is copied
///<
template <typename R>
void mqi::io::save_to_bin(const mqi::scorer<R> *src, const R scale,
                          const std::string &filepath,
                          const std::string &filename) {
  /// create a copy using valarray and apply scale

  unsigned int nnz = 0;
  std::vector<mqi::key_t> key1;
  std::vector<mqi::key_t> key2;
  std::vector<double> value;
  key1.clear();
  key2.clear();
  value.clear();
  for (int ind = 0; ind < src->max_capacity_; ind++) {
    if (src->data_[ind].key1 != mqi::empty_pair &&
        src->data_[ind].key2 != mqi::empty_pair && src->data_[ind].value > 0) {
      key1.push_back(src->data_[ind].key1);
      key2.push_back(src->data_[ind].key2);
      value.push_back(src->data_[ind].value * scale);
    }
  }

  printf("length %lu %lu %lu\n", key1.size(), key2.size(), value.size());

  /// open out stream
  std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                         std::ios::out | std::ios::binary);
  if (!fid_key1)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw"
              << std::endl;

  /// write to a file
  fid_key1.write(reinterpret_cast<const char *>(&key1.data()[0]),
                 key1.size() * sizeof(mqi::key_t));
  fid_key1.close();

  std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                         std::ios::out | std::ios::binary);
  if (!fid_key2)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw"
              << std::endl;

  /// write to a file
  fid_key2.write(reinterpret_cast<const char *>(&key2.data()[0]),
                 key2.size() * sizeof(mqi::key_t));
  fid_key2.close();

  std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                        std::ios::out | std::ios::binary);
  if (!fid_bin)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw"
              << std::endl;

  /// write to a file
  fid_bin.write(reinterpret_cast<const char *>(&value.data()[0]),
                value.size() * sizeof(double));
  fid_bin.close();
}

///< Function to write array into file
///< src: array and this array is copied
///<
template <typename R>
void mqi::io::save_to_bin(const R *src, const R scale,
                          const std::string &filepath,
                          const std::string &filename, const uint32_t length) {
  /// create a copy using valarray and apply scale
  std::valarray<R> dest(src, length);
  munmap(&dest, length * sizeof(R));
  dest *= scale;
  /// open out stream
  std::ofstream fid_bin(filepath + "/" + filename + ".raw",
                        std::ios::out | std::ios::binary);
  if (!fid_bin)
    std::cout << "Cannot write :" << filepath + "/" + filename + ".raw"
              << std::endl;

  /// write to a file
  fid_bin.write(reinterpret_cast<const char *>(&dest[0]), length * sizeof(R));
  fid_bin.close();
}

///< Function to write key values into file
///< src: array and this array is copied
///<
template <typename R>
void mqi::io::save_to_bin(const mqi::key_value *src, const R scale,
                          uint32_t max_capacity, const std::string &filepath,
                          const std::string &filename) {
  /// create a copy using valarray and apply scale

  unsigned int nnz = 0;
  std::vector<mqi::key_t> key1;
  std::vector<mqi::key_t> key2;
  std::vector<R> value;
  key1.clear();
  key2.clear();
  value.clear();
  for (int ind = 0; ind < max_capacity; ind++) {
    if (src[ind].key1 != mqi::empty_pair && src[ind].key2 != mqi::empty_pair &&
        src[ind].value > 0) {
      key1.push_back(src[ind].key1);
      key2.push_back(src[ind].key2);
      value.push_back(src[ind].value * scale);
    }
  }

  printf("length %lu %lu %lu\n", key1.size(), key2.size(), value.size());
  /// open out stream
  std::ofstream fid_key1(filepath + "/" + filename + "_key1.raw",
                         std::ios::out | std::ios::binary);
  if (!fid_key1)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_key1.raw"
              << std::endl;

  /// write to a file
  fid_key1.write(reinterpret_cast<const char *>(&key1.data()[0]),
                 key1.size() * sizeof(mqi::key_t));
  fid_key1.close();

  std::ofstream fid_key2(filepath + "/" + filename + "_key2.raw",
                         std::ios::out | std::ios::binary);
  if (!fid_key2)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_key2.raw"
              << std::endl;

  /// write to a file
  fid_key2.write(reinterpret_cast<const char *>(&key2.data()[0]),
                 key2.size() * sizeof(mqi::key_t));
  fid_key2.close();

  std::ofstream fid_bin(filepath + "/" + filename + "_value.raw",
                        std::ios::out | std::ios::binary);
  if (!fid_bin)
    std::cout << "Cannot write :" << filepath + "/" + filename + "_value.raw"
              << std::endl;

  /// write to a file
  fid_bin.write(reinterpret_cast<const char *>(&value.data()[0]),
                value.size() * sizeof(R));
  fid_bin.close();
}

///< Function to write key values into file
///< src: array and this array is copied
///<

template <typename R>
void mqi::io::save_to_npz(const mqi::scorer<R> *src, const R scale,
                          const std::string &filepath,
                          const std::string &filename,
                          mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots) {
  uint32_t vol_size;
  vol_size = dim.x * dim.y * dim.z;

  /// create a copy using valarray and apply scale
  const std::string name_a = "indices.npy", name_b = "indptr.npy",
                    name_c = "shape.npy", name_d = "data.npy",
                    name_e = "format.npy";
  std::vector<double> *value_vec = new std::vector<double>[num_spots];
  std::vector<mqi::key_t> *vox_vec = new std::vector<mqi::key_t>[num_spots];
  std::vector<double> data_vec;
  std::vector<uint32_t> indices_vec;
  std::vector<uint32_t> indptr_vec;
  mqi::key_t vox_ind, spot_ind;
  double value;
  int spot_start = 0, spot_end = 0;
  int vox_in_spot[num_spots];
  std::vector<double>::iterator it_data;
  std::vector<uint32_t>::iterator it_ind;
  std::vector<mqi::key_t>::iterator it_spot;
  int vox_count;
  printf("save_to_npz\n");

  printf("scan start %d\n", src->max_capacity_);
  for (int ind = 0; ind < src->max_capacity_; ind++) {
    if (src->data_[ind].key1 != mqi::empty_pair &&
        src->data_[ind].key2 != mqi::empty_pair) {
      vox_count = 0;
      vox_ind = src->data_[ind].key1;
      spot_ind = src->data_[ind].key2;
      assert(vox_ind >= 0 && vox_ind < vol_size);
      value = src->data_[ind].value;
      value_vec[spot_ind].push_back(value * scale);
      vox_vec[spot_ind].push_back(vox_ind);
    }
  }

  vox_count = 0;
  indptr_vec.push_back(vox_count);
  for (int ii = 0; ii < num_spots; ii++) {
    data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
    indices_vec.insert(indices_vec.end(), vox_vec[ii].begin(),
                       vox_vec[ii].end());
    vox_count += vox_vec[ii].size();
    indptr_vec.push_back(vox_count);
  }

  printf("scan done %lu %lu %lu\n", data_vec.size(), indices_vec.size(),
         indptr_vec.size());
  printf("%d %d\n", vol_size, num_spots);

  uint32_t shape[2] = {num_spots, vol_size};
  std::string format = "csr";
  size_t size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
         size_d = data_vec.size(), size_e = 3;

  uint32_t *indices = new uint32_t[indices_vec.size()];
  uint32_t *indptr = new uint32_t[indptr_vec.size()];
  double *data = new double[data_vec.size()];
  std::copy(indices_vec.begin(), indices_vec.end(), indices);
  std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
  std::copy(data_vec.begin(), data_vec.end(), data);
  printf("%lu\n", size_b);
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a,
                    "w");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e,
                    "a");
}

template <typename R>
void mqi::io::save_to_npz2(const mqi::scorer<R> *src, const R scale,
                           const std::string &filepath,
                           const std::string &filename,
                           mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots) {
  uint32_t vol_size;
  vol_size = src->roi_->get_mask_size();
  /// create a copy using valarray and apply scale
  const std::string name_a = "indices.npy", name_b = "indptr.npy",
                    name_c = "shape.npy", name_d = "data.npy",
                    name_e = "format.npy";

  std::vector<double> *value_vec = new std::vector<double>[vol_size];
  std::vector<mqi::key_t> *spot_vec = new std::vector<mqi::key_t>[vol_size];
  std::vector<double> data_vec;
  std::vector<uint32_t> indices_vec;
  std::vector<uint32_t> indptr_vec;
  mqi::key_t vox_ind, spot_ind;
  double value;
  int spot_start = 0, spot_end = 0;
  std::vector<double>::iterator it_data;
  std::vector<uint32_t>::iterator it_ind;
  std::vector<mqi::key_t>::iterator it_spot;
  int spot_count;
  printf("save_to_npz\n");

  printf("scan start %d\n", src->max_capacity_);
  for (int ind = 0; ind < src->max_capacity_; ind++) {
    if (src->data_[ind].key1 != mqi::empty_pair &&
        src->data_[ind].key2 != mqi::empty_pair) {
      vox_ind = src->data_[ind].key1;
      vox_ind = src->roi_->get_mask_idx(vox_ind);
      if (vox_ind < 0) {
        printf("is this right?\n");
        continue;
      }
      spot_ind = src->data_[ind].key2;
      assert(vox_ind >= 0 && vox_ind < vol_size);
      value = src->data_[ind].value;
      assert(value > 0);
      value_vec[vox_ind].push_back(value * scale);
      spot_vec[vox_ind].push_back(spot_ind);
    }
  }
  printf("Sorting start\n");
  for (int ind = 0; ind < vol_size; ind++) {
    if (spot_vec[ind].size() > 1) {
      std::vector<int> sort_ind(spot_vec[ind].size());
      std::iota(sort_ind.begin(), sort_ind.end(), 0);
      sort(sort_ind.begin(), sort_ind.end(),
           [&](int i, int j) { return spot_vec[ind][i] < spot_vec[ind][j]; });
      std::vector<double> sorted_value(spot_vec[ind].size());
      std::vector<mqi::key_t> sorted_spot(spot_vec[ind].size());
      for (int sorted_ind = 0; sorted_ind < spot_vec[ind].size();
           sorted_ind++) {
        sorted_value[sorted_ind] = value_vec[ind][sort_ind[sorted_ind]];
        sorted_spot[sorted_ind] = spot_vec[ind][sort_ind[sorted_ind]];
      }
      spot_vec[ind] = sorted_spot;
      value_vec[ind] = sorted_value;
    }
  }

  spot_count = 0;
  indptr_vec.push_back(spot_count);
  for (int ii = 0; ii < vol_size; ii++) {
    data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
    indices_vec.insert(indices_vec.end(), spot_vec[ii].begin(),
                       spot_vec[ii].end());
    spot_count += spot_vec[ii].size();
    indptr_vec.push_back(spot_count);
  }

  uint32_t shape[2] = {vol_size, num_spots};
  std::string format = "csr";
  size_t size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
         size_d = data_vec.size(), size_e = 3;

  uint32_t *indices = new uint32_t[indices_vec.size()];
  uint32_t *indptr = new uint32_t[indptr_vec.size()];
  double *data = new double[data_vec.size()];
  std::copy(indices_vec.begin(), indices_vec.end(), indices);
  std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
  std::copy(data_vec.begin(), data_vec.end(), data);
  printf("%lu\n", size_b);
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a,
                    "w");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e,
                    "a");
}

template <typename R>
void mqi::io::save_to_npz(const mqi::scorer<R> *src, const R scale,
                          const std::string &filepath,
                          const std::string &filename,
                          mqi::vec3<mqi::ijk_t> dim, uint32_t num_spots,
                          R *time_scale, R threshold) {
  uint32_t vol_size;
  vol_size = dim.x * dim.y * dim.z;
  /// create a copy using valarray and apply scale
  const std::string name_a = "indices.npy", name_b = "indptr.npy",
                    name_c = "shape.npy", name_d = "data.npy",
                    name_e = "format.npy";
  std::vector<double> value_vec[num_spots];
  std::vector<mqi::key_t> vox_vec[num_spots];
  std::vector<double> data_vec;
  std::vector<uint32_t> indices_vec;
  std::vector<uint32_t> indptr_vec;
  mqi::key_t vox_ind, spot_ind;
  double value;
  int spot_start = 0, spot_end = 0;
  int vox_in_spot[num_spots];
  std::vector<double>::iterator it_data;
  std::vector<uint32_t>::iterator it_ind;
  std::vector<mqi::key_t>::iterator it_spot;
  int vox_count;
  printf("save_to_npz\n");
  for (int ind = 0; ind < num_spots; ind++) {
    vox_in_spot[ind] = 0;
  }
  printf("scan start %d\n", src->max_capacity_);
  for (int ind = 0; ind < src->max_capacity_; ind++) {
    if (src->data_[ind].key1 != mqi::empty_pair &&
        src->data_[ind].key2 != mqi::empty_pair) {
      vox_count = 0;
      vox_ind = src->data_[ind].key1;
      spot_ind = src->data_[ind].key2;
      assert(vox_ind >= 0 && vox_ind < vol_size);
      value = src->data_[ind].value;
      value *= scale;
      value -= 2 * threshold;
      if (value < 0)
        value = 0;
      value /= time_scale[spot_ind];
      value_vec[spot_ind].push_back(value);
      vox_vec[spot_ind].push_back(vox_ind);
    }
  }

  vox_count = 0;
  indptr_vec.push_back(vox_count);
  for (int ii = 0; ii < num_spots; ii++) {
    data_vec.insert(data_vec.end(), value_vec[ii].begin(), value_vec[ii].end());
    indices_vec.insert(indices_vec.end(), vox_vec[ii].begin(),
                       vox_vec[ii].end());
    vox_count += vox_vec[ii].size();
    indptr_vec.push_back(vox_count);
  }
  printf("scan done %lu %lu %lu\n", data_vec.size(), indices_vec.size(),
         indptr_vec.size());
  printf("%d %d\n", vol_size, num_spots);

  uint32_t shape[2] = {num_spots, vol_size};
  std::string format = "csr";
  size_t size_a = indices_vec.size(), size_b = indptr_vec.size(), size_c = 2,
         size_d = data_vec.size(), size_e = 3;

  uint32_t *indices = new uint32_t[indices_vec.size()];
  uint32_t *indptr = new uint32_t[indptr_vec.size()];
  double *data = new double[data_vec.size()];
  std::copy(indices_vec.begin(), indices_vec.end(), indices);
  std::copy(indptr_vec.begin(), indptr_vec.end(), indptr);
  std::copy(data_vec.begin(), data_vec.end(), data);
  printf("%lu\n", size_b);
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_a, indices, size_a,
                    "w");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_b, indptr, size_b,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_c, shape, size_c,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_d, data, size_d,
                    "a");
  mqi::io::save_npz(filepath + "/" + filename + ".npz", name_e, format, size_e,
                    "a");
}

template <typename R>
void mqi::io::save_to_mhd(const mqi::node_t<R> *children, const double *src,
                          const R scale, const std::string &filepath,
                          const std::string &filename, const uint32_t length) {
  ///< TODO: this works only for two depth world
  ///< TODO: dx, dy, and dz calculation works only for AABB
  float dx = children->geo[0].get_x_edges()[1];
  dx -= children->geo[0].get_x_edges()[0];
  float dy = children->geo[0].get_y_edges()[1];
  dy -= children->geo[0].get_y_edges()[0];
  float dz = children->geo[0].get_z_edges()[1];
  dz -= children->geo[0].get_z_edges()[0];
  float x0 = children->geo[0].get_x_edges()[0];
  x0 += children->geo[0].get_x_edges()[0];
  x0 /= 2.0;
  float y0 = children->geo[0].get_y_edges()[0];
  y0 += children->geo[0].get_y_edges()[0];
  y0 /= 2.0;
  float z0 = children->geo[0].get_z_edges()[0];
  z0 += children->geo[0].get_z_edges()[0];
  z0 /= 2.0;
  std::ofstream fid_header(filepath + "/" + filename + ".mhd", std::ios::out);
  if (!fid_header) {
    std::cout << "Cannot open file!" << std::endl;
  }
  fid_header << "ObjectType = Image\n";
  fid_header << "NDims = 3\n";
  fid_header << "BinaryData = True\n";
  fid_header << "BinaryDataByteOrderMSB = False\n"; // True for big endian,
                                                    // False for little endian
  fid_header << "CompressedData = False\n";
  fid_header << "TransformMatrix 1 0 0 0 1 0 0 0 1\n";
  fid_header << "Offset " << x0 << " " << y0 << " " << z0 << std::endl;
  fid_header << "CenterOfRotation 0 0 0\n";
  fid_header << "AnatomicOrientation = RAI\n";
  fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
             << children->geo[0].get_nxyz().y << " "
             << children->geo[0].get_nxyz().z << "\n";
  ///< TODO: if R is double, MET_FLOAT should be MET_DOUBLE
  fid_header << "ElementType = MET_DOUBLE\n";

  fid_header << "ElementSpacing = " << dx << " " << dy << " " << dz << "\n";
  fid_header << "ElementDataFile = " << filename << ".raw" << "\n";
  fid_header.close();
  if (!fid_header.good()) {
    std::cout << "Error occurred at writing time!" << std::endl;
  }
  std::valarray<double> dest(src, length);
  munmap(&dest, length * sizeof(double));
  dest *= scale;
  std::ofstream fid_raw(filepath + "/" + filename + ".raw",
                        std::ios::out | std::ios::binary);
  if (!fid_raw) {
    std::cout << "Cannot open file!" << std::endl;
  }
  fid_raw.write(reinterpret_cast<const char *>(&dest[0]),
                length * sizeof(double));

  fid_raw.close();
  if (!fid_raw.good()) {
    std::cout << "Error occurred at writing time!" << std::endl;
  }
}

template <typename R>
void mqi::io::save_to_mha(const mqi::node_t<R> *children, const double *src,
                          const R scale, const std::string &filepath,
                          const std::string &filename, const uint32_t length) {
  ///< TODO: this works only for two depth world
  ///< TODO: dx, dy, and dz calculation works only for AABB
  float dx = children->geo[0].get_x_edges()[1];
  dx -= children->geo[0].get_x_edges()[0];
  float dy = children->geo[0].get_y_edges()[1];
  dy -= children->geo[0].get_y_edges()[0];
  float dz = children->geo[0].get_z_edges()[1];
  dz -= children->geo[0].get_z_edges()[0];
  float x0 = children->geo[0].get_x_edges()[0] + dx * 0.5;
  float y0 = children->geo[0].get_y_edges()[0] + dy * 0.5;
  float z0 = children->geo[0].get_z_edges()[0] + dz * 0.5;
  std::cout << "x0 " << std::setprecision(9) << x0 << " y0 " << y0 << " z0 "
            << z0 << std::endl;
  std::valarray<double> dest(src, length);
  munmap(&dest, length * sizeof(double));
  dest *= scale;
  std::ofstream fid_header(filepath + "/" + filename + ".mha", std::ios::out);
  if (!fid_header) {
    std::cout << "Cannot open file!" << std::endl;
  }
  fid_header << "ObjectType = Image\n";
  fid_header << "NDims = 3\n";
  fid_header << "BinaryData = True\n";
  fid_header << "BinaryDataByteOrderMSB = False\n"; // True for big endian,
                                                    // False for little endian
  fid_header << "CompressedData = False\n";
  fid_header << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n";
  fid_header << "Origin = " << std::setprecision(9) << x0 << " " << y0 << " "
             << z0 << "\n";
  fid_header << "CenterOfRotation = 0 0 0\n";
  fid_header << "AnatomicOrientation = RAI\n";
  fid_header << "DimSize = " << children->geo[0].get_nxyz().x << " "
             << children->geo[0].get_nxyz().y << " "
             << children->geo[0].get_nxyz().z << "\n";
  ///< TODO: if R is double, MET_FLOAT should be MET_DOUBLE
  fid_header << "ElementType = MET_DOUBLE\n";
  fid_header << "HeaderSize = -1\n";
  fid_header << "ElementSpacing = " << std::setprecision(9) << dx << " " << dy
             << " " << dz << "\n";
  fid_header << "ElementDataFile = LOCAL\n";
  fid_header.write(reinterpret_cast<const char *>(&dest[0]),
                   length * sizeof(double));
  fid_header.close();
  if (!fid_header.good()) {
    std::cout << "Error occurred at writing time!" << std::endl;
  }
}


template <typename R>
void mqi::io::save_to_dcm(const mqi::node_t<R> *child, const double *src,
                          const R scale, const std::string &filepath,
                          const std::string &filename, const uint32_t length,
                          const mqi::dicom_t &dcm_info, bool two_cm_mode) {
#if DCMTK_FOUND
  // Full DCMTK implementation
  // Input validation
  if (!child) {
    std::cerr << "Error: child node is null" << std::endl;
    return;
  }
  if (!src) {
    std::cerr << "Error: source data pointer is null" << std::endl;
    return;
  }
  if (length == 0) {
    std::cerr << "Error: data length is zero" << std::endl;
    return;
  }
  if (dcm_info.plan_name.empty()) {
    std::cerr << "Error: plan name is empty" << std::endl;
    return;
  }

  DcmFileFormat fileformat;
  DcmDataset *dataset = fileformat.getDataset();

  // Helper lambda to add or insert a new element
  auto put_string = [&](const DcmTagKey &tag, const std::string &value) {
    dataset->putAndInsertString(tag, value.c_str());
  };

  // Read the source RTPLAN file to copy metadata
  DcmFileFormat plan_ff;
  OFCondition status = plan_ff.loadFile(dcm_info.plan_name.c_str());
  if (status.bad()) {
    std::cerr << "Error: cannot read RTPLAN file: " << dcm_info.plan_name
              << " (" << status.text() << ")" << std::endl;
    return;
  }
  DcmDataset *plan_dataset = plan_ff.getDataset();
  if (!plan_dataset) {
    std::cerr << "Error: cannot get dataset from RTPLAN file" << std::endl;
    return;
  }

  // Copy patient and study information
  OFString patient_name, patient_id, study_instance_uid, series_instance_uid_plan, frame_of_reference_uid;
  if (plan_dataset->findAndGetOFString(DCM_PatientName, patient_name).bad() ||
      plan_dataset->findAndGetOFString(DCM_PatientID, patient_id).bad() ||
      plan_dataset->findAndGetOFString(DCM_StudyInstanceUID, study_instance_uid).bad() ||
      plan_dataset->findAndGetOFString(DCM_FrameOfReferenceUID, frame_of_reference_uid).bad()) {
    std::cerr << "Warning: Some patient/study information could not be read from RTPLAN" << std::endl;
  }
  plan_dataset->findAndGetOFString(DCM_SeriesInstanceUID, series_instance_uid_plan);

  put_string(DCM_PatientName, patient_name.c_str());
  put_string(DCM_PatientID, patient_id.c_str());
  put_string(DCM_StudyInstanceUID, study_instance_uid.c_str());
  put_string(DCM_FrameOfReferenceUID, frame_of_reference_uid.c_str());

  // SOP Class and Instance UID
  put_string(DCM_SOPClassUID, UID_RTDoseStorage);
  char new_uid[100];
  dcmGenerateUniqueIdentifier(new_uid, SITE_INSTANCE_UID_ROOT);
  put_string(DCM_SOPInstanceUID, new_uid);
  dcmGenerateUniqueIdentifier(new_uid, SITE_INSTANCE_UID_ROOT);
  put_string(DCM_SeriesInstanceUID, new_uid);


  // Dose information
  put_string(DCM_DoseUnits, "GY");
  put_string(DCM_DoseType, "PHYSICAL");
  put_string(DCM_DoseComment, "Generated by MOQUI Monte Carlo dose calculation");

  // Additional required RT Dose tags
  put_string(DCM_Modality, "RTDOSE");
  put_string(DCM_Manufacturer, "MOQUI Monte Carlo System");
  put_string(DCM_ManufacturerModelName, "v1.0.8");
  put_string(DCM_SeriesDescription, "Monte Carlo Calculated Dose");

  // Dose Grid Scaling: find max dose to set the scale
  double max_dose = 0.0;
  for (uint32_t i = 0; i < length; ++i) {
    if (src[i] > max_dose) {
      max_dose = src[i];
    }
  }

  // Handle zero-dose scenario
  double dose_grid_scaling;
  if (max_dose <= 0.0) {
    std::cerr << "Warning: Maximum dose is zero or negative. Setting scaling factor to minimum value." << std::endl;
    dose_grid_scaling = 1e-6; // Minimum scaling factor to avoid division by zero
  } else {
    dose_grid_scaling = (max_dose * scale) / 65535.0;
  }

  // Ensure minimum scaling factor to avoid precision issues
  if (dose_grid_scaling < 1e-6) {
    dose_grid_scaling = 1e-6;
  }

  dataset->putAndInsertFloat64(DCM_DoseGridScaling, dose_grid_scaling);

  // Geometry Information
  const auto &geo = child->geo[0];
  const auto &nxyz = geo.get_nxyz();
  float dx = geo.get_x_edges()[1] - geo.get_x_edges()[0];
  float dy = geo.get_y_edges()[1] - geo.get_y_edges()[0];
  float z0 = geo.get_z_edges()[0];

  put_string(DCM_ImageOrientationPatient, "1\\0\\0\\0\\1\\0");
  dataset->putAndInsertUint16(DCM_Rows, nxyz.y);
  dataset->putAndInsertUint16(DCM_Columns, nxyz.x);

  char buffer[64];
  sprintf(buffer, "%f\\%f", dy, dx);
  put_string(DCM_PixelSpacing, buffer);

  if (two_cm_mode) {
      dataset->putAndInsertUint16(DCM_NumberOfFrames, 1);
      int z_index = -1;

      // Bounds checking for 2cm mode
      if (nxyz.z < 1) {
          std::cerr << "Error: Insufficient number of slices for 2CM mode" << std::endl;
          return;
      }

      // In TwoCentimeterMode, we expect a single slice at the 2cm position
      // Use the middle slice (index 0) for single-slice geometry
      if (nxyz.z == 1) {
          z_index = 0;
      } else {
          // Search for slice closest to 20mm depth
          float target_depth = 20.0f;
          float min_distance = std::numeric_limits<float>::max();
          for(int i = 0; i < nxyz.z; ++i) {
              float slice_center = (geo.get_z_edges()[i] + geo.get_z_edges()[i+1]) / 2.0f;
              float distance = std::abs(slice_center - target_depth);
              if (distance < min_distance) {
                  min_distance = distance;
                  z_index = i;
              }
          }
      }

      if (z_index == -1) {
          std::cerr << "Error: could not determine appropriate slice for 2CM mode" << std::endl;
          return;
      }

      // Bounds check for array access
      if (z_index >= nxyz.z || z_index < 0) {
          std::cerr << "Error: Invalid slice index calculated: " << z_index << std::endl;
          return;
      }

      sprintf(buffer, "%f\\%f\\%f", geo.get_x_edges()[0], geo.get_y_edges()[0], geo.get_z_edges()[z_index]);
      put_string(DCM_ImagePositionPatient, buffer);

      // Calculate actual slice thickness instead of hardcoded value
      float slice_thickness = geo.get_z_edges()[z_index + 1] - geo.get_z_edges()[z_index];
      if (slice_thickness <= 0.0f) {
          std::cerr << "Warning: Invalid slice thickness calculated: " << slice_thickness << std::endl;
          slice_thickness = 1.0f; // Fallback value
      }
      sprintf(buffer, "%f", slice_thickness);
      put_string(DCM_SliceThickness, buffer);

      // Set GridFrameOffsetVector for single frame (should be "0")
      put_string(DCM_GridFrameOffsetVector, "0");

      // Use RAII for memory management
      uint32_t slice_size = nxyz.x * nxyz.y;
      if (z_index * slice_size >= length) {
          std::cerr << "Error: Calculated slice index exceeds data length" << std::endl;
          return;
      }

      std::vector<uint16_t> pixel_data(slice_size);
      for (uint32_t i = 0; i < slice_size; ++i) {
          pixel_data[i] = static_cast<uint16_t>((src[z_index * slice_size + i] * scale) / dose_grid_scaling);
      }
      dataset->putAndInsertUint16Array(DCM_PixelData, pixel_data.data(), slice_size);

  } else {
      dataset->putAndInsertUint16(DCM_NumberOfFrames, nxyz.z);
      sprintf(buffer, "%f\\%f\\%f", geo.get_x_edges()[0], geo.get_y_edges()[0], z0);
      put_string(DCM_ImagePositionPatient, buffer);

      OFString grid_frame_offset_vector = "0";
      float current_offset = 0.0f;
      for (int i = 0; i < nxyz.z - 1; ++i) {
          float dz = geo.get_z_edges()[i+1] - geo.get_z_edges()[i];
          current_offset += dz;
          sprintf(buffer, "\\%f", current_offset);
          grid_frame_offset_vector.append(buffer);
      }
      put_string(DCM_GridFrameOffsetVector, grid_frame_offset_vector.c_str());

      // Use RAII for memory management
      std::vector<uint16_t> pixel_data(length);
      for (uint32_t i = 0; i < length; ++i) {
          pixel_data[i] = static_cast<uint16_t>((src[i] * scale) / dose_grid_scaling);
      }
      dataset->putAndInsertUint16Array(DCM_PixelData, pixel_data.data(), length);
  }

  // Pixel Data
  put_string(DCM_PixelRepresentation, "0"); // Unsigned integer
  put_string(DCM_BitsAllocated, "16");
  put_string(DCM_BitsStored, "16");
  put_string(DCM_HighBit, "15");


  // Save file
  std::string output_file = filepath + "/" + filename + ".dcm";
  OFCondition save_status = fileformat.saveFile(output_file.c_str(), EXS_LittleEndianExplicit);
  if (save_status.bad()) {
    std::cerr << "Error: cannot write DICOM file " << output_file << " (" << save_status.text() << ")" << std::endl;
  }
#else
  // Fallback implementation when DCMTK is not available
  std::cerr << "=================================================================" << std::endl;
  std::cerr << "ERROR: DCMTK library not found - DICOM RT Dose output disabled" << std::endl;
  std::cerr << "=================================================================" << std::endl;
  std::cerr << "To enable DICOM output functionality, please install DCMTK:" << std::endl;
  std::cerr << "  Ubuntu/Debian: sudo apt-get install libdcmtk-dev dcmtk" << std::endl;
  std::cerr << "  CentOS/RHEL:   sudo yum install dcmtk-devel" << std::endl;
  std::cerr << "  macOS:         brew install dcmtk" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Alternatively, use a different output format:" << std::endl;
  std::cerr << "  - OutputFormat mhd (MetaImage format)" << std::endl;
  std::cerr << "  - OutputFormat mha (MetaImage format)" << std::endl;
  std::cerr << "  - OutputFormat raw (raw binary format)" << std::endl;
  std::cerr << "=================================================================" << std::endl;

  // Save as MHD format as fallback
  std::cout << "Falling back to MHD format output..." << std::endl;
  mqi::io::save_to_mhd<R>(child, src, scale, filepath, filename, length);
#endif
}

#endif
