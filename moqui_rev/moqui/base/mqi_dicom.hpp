#ifndef MQI_DICOM_HPP
#define MQI_DICOM_HPP

#include "mqi_vec.hpp"
#include "mqi_ct.hpp"
#include <string>
#include <vector>
#include "gdcmDirectory.h"

namespace mqi {
struct dicom_t {
  mqi::vec3<ijk_t> dim_;     // number of voxels
  mqi::vec3<ijk_t> org_dim_; // number of voxels
  float dx = -1;
  float dy = -1;
  float *org_dz;
  float *dz;
  uint16_t num_vol = 0;
  uint16_t nfiles = 0;
  uint16_t n_plan = 0;
  uint16_t n_dose = 0;
  uint16_t n_struct = 0;
  float *xe = nullptr;
  float *ye = nullptr;
  float *ze = nullptr;
  float *org_xe = nullptr;
  float *org_ye = nullptr;
  float *org_ze = nullptr;
  gdcm::Directory::FilenamesType plan_list;
  gdcm::Directory::FilenamesType dose_list;
  gdcm::Directory::FilenamesType struct_list;
  gdcm::Directory::FilenamesType ct_list;
  std::string plan_name = "";
  std::string struct_name = "";
  std::string dose_name = "";
  mqi::ct<phsp_t> *ct;
  mqi::vec3<float> image_center;
  mqi::vec3<size_t> dose_dim;
  mqi::vec3<float> dose_pos0;
  float dose_dx;
  float dose_dy;
  float *dose_dz;
  mqi::vec3<uint16_t> clip_shift_;
  uint8_t *body_contour;
};
}
#endif