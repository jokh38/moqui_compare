#ifndef MQI_TPS_RAY_ENV_HPP
#define MQI_TPS_RAY_ENV_HPP

#include <moqui/base/environments/mqi_xenvironment.hpp>
#include <moqui/base/scorers/mqi_scorer_energy_deposit.hpp>

#include "gdcmAttribute.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGlobal.h"
#include "gdcmIPPSorter.h"
#include "gdcmImage.h"
#include "gdcmReader.h"
#include "gdcmScanner.h"
#include "gdcmSorter.h"
#include "gdcmStringFilter.h"
#include "gdcmTag.h"
#include "gdcmTesting.h"
#include <cassert>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <set>
#include <moqui/base/materials/mqi_patient_materials.hpp>
#include <moqui/base/mqi_aperture.hpp>
#include <moqui/base/mqi_aperture3d.hpp>
#include <moqui/base/mqi_distributions.hpp>
#include <moqui/base/mqi_file_handler.hpp>
#include <moqui/base/mqi_io.hpp>
#include <moqui/base/mqi_math.hpp>
#include <moqui/base/mqi_rangeshifter.hpp>
#include <moqui/base/mqi_roi.hpp>
#include <moqui/base/mqi_threads.hpp>
#include <moqui/base/mqi_treatment_session.hpp>
#include <moqui/base/scorers/mqi_scorer_energy_deposit.hpp>
#include <valarray>

namespace mqi
{
struct dicom_t {
    mqi::vec3<ijk_t>               dim_;       //number of voxels
    mqi::vec3<ijk_t>               org_dim_;   //number of voxels
    float                          dx = -1;
    float                          dy = -1;
    float*                         org_dz;
    float*                         dz;
    uint16_t                       num_vol  = 0;
    uint16_t                       nfiles   = 0;
    uint16_t                       n_plan   = 0;
    uint16_t                       n_dose   = 0;
    uint16_t                       n_struct = 0;
    float*                         xe       = nullptr;
    float*                         ye       = nullptr;
    float*                         ze       = nullptr;
    float*                         org_xe   = nullptr;
    float*                         org_ye   = nullptr;
    float*                         org_ze   = nullptr;
    gdcm::Directory::FilenamesType plan_list;
    gdcm::Directory::FilenamesType dose_list;
    gdcm::Directory::FilenamesType struct_list;
    gdcm::Directory::FilenamesType ct_list;
    std::string                    plan_name   = "";
    std::string                    struct_name = "";
    std::string                    dose_name   = "";
    mqi::ct<phsp_t>*               ct;
    mqi::vec3<float>               image_center;
    mqi::vec3<size_t>              dose_dim;
    mqi::vec3<float>               dose_pos0;
    float                          dose_dx;
    float                          dose_dy;
    float*                         dose_dz;
    mqi::vec3<uint16_t>            clip_shift_;
    uint8_t*                       body_contour;
};



template<typename R>
class tps_env : public x_environment<R>
{
public:
    std::string input_filename   = "";
    std::string delimeter        = " ";
    int         master_seed      = 0;
    std::string machine_name     = "";
    bool        score_to_ct_grid = true;
    std::string output_path      = "";
    std::string output_format =
      "";   /// currently support mhd, other than mhd are considered as binary
    /// Scorer parameters
    std::string   scorer_string;
    mqi::scorer_t scorer_type;
    bool          score_variance = false;
    std::string   source_type    = "FluenceMap";
    /// Simulation parameters
    mqi::sim_type_t            sim_type;
    std::vector<int>           beam_numbers;
    std::string                parent_dir = "";
    std::string                dicom_dir  = "";
    std::string                logfile_dir = ""; // Log file dir added in 2023-11-01

    int selectedGantryNumber = 2; // Basic gantry is G2
    bool usingPhantomGeo = false; // to use phantom environment, added in 2024-04-11
    bool twoCentimeterMode = false; // to use patient-specific QA
    int phantomDimX = 400;
    int phantomDimY = 400;
    int phantomDimZ = 400;
    int phantomUnitX = 1;
    int phantomUnitY = 1;
    int phantomUnitZ = 1;
    float phantomPositionX = -200.f;
    float phantomPositionY = -200.f;
    float phantomPositionZ = -200.f;

    std::vector<std::string>   parameters_total;
    std::vector<std::string>   mask_filenames;
    bool                       scoring_mask      = false;
    bool                       overwrite_results = false;
    bool                       use_absolute_path;
    size_t                     max_histories_per_batch;
    bool                       memory_save_mode;
    bool                       save_scorer_map;
    std::string                scorer_map_prefix;
    dicom_t                    dcm_;
    logfiles_t                 log_file;
    int16_t*                   ct_data;
    mqi::treatment_session<R>* tx;
    uint16_t                   bnb                   = 0;
    float                      sid                   = 0.0;
    float                      particles_per_history = -1.0;
    std::string                beam_prefix;
    density_t*                 stopping_power;
    float                      max_let_in_water;
    int                        aperture_ind  = -1;
    mqi::aperture_type_t       aperture_type = mqi::VOLUME;
    std::vector<float>         scorer_voxel_size;
    bool                       ct_clipping;
    int                        verbosity;
    std::string                body_contour_name;
    bool                       read_structure;
    uint32_t                   scorer_size;
    uint32_t                   scorer_capacity;
    bool                       reshape_output = false;
    bool                       sparse_output  = false;
    //    std::default_random_engine beam_rng;

public:
    CUDA_HOST
    tps_env(const std::string input_name) : x_environment<R>() {
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        std::cout << "MOQUI SMC Treatment machine model and log-file based MC" << std::endl;
        std::cout << "Modified by Chanil Jeon of Sungkyunkwan University in 2024" << std::endl;
        std::cout << "Version : v1.0.8 (2024-08-27)" << std::endl;
        std::cout << "Starting process of loading basic information.." << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        struct stat info;
        std::string delimeter = " ";
        input_filename        = input_name;
        mqi::file_parser parser(input_filename, delimeter);
        /// Global parameters
        this->gpu_id = parser.get_int("GPUID", 0);
        std::cout << "Reading GPU ID.. : Selected GPU ID --> " << this->gpu_id << std::endl; 
#if defined(__CUDACC__)
        cudaSetDevice(this->gpu_id);
#endif
        master_seed = parser.get_int("RandomSeed", -1);
        std::cout << "Reading seeds.. : Inserted seed --> " << master_seed << std::endl;
        if (master_seed < -1) std::cout << "Reading seeds.. : Using negative value generates current time based random seed." << std::endl;
        use_absolute_path = parser.get_bool("UseAbsolutePath", false);

        this->num_total_threads = parser.get_int("TotalThreads", -1);
        beam_prefix             = parser.get_string("BeamPrefix", "beam");
        max_histories_per_batch = parser.get_int("MaxHistoriesPerBatch", 0);
        //        std::string aperture_string = parser.get_string("ApertureType", "VOLUME");
        //        aperture_type           = parser.string_to_aperture_type(aperture_string);

        // -----------------------------------------------------------------------------------------------
        /// Data directories
        if (use_absolute_path) {
            this->parent_dir = parser.get_string("ParentDir", "");
            this->dicom_dir  = parser.get_string("DicomPath", "");
            this->dicom_dir  = parser.get_string("logFilePath", ""); // Log file dir added in 2023-11-01 by Chanil Jeon
        } else {
            this->parent_dir = parser.get_string("ParentDir", "");
            this->dicom_dir  = this->parent_dir + "/" + parser.get_string("DicomDir", "");
            this->logfile_dir  = this->parent_dir + "/" + parser.get_string("logFilePath", ""); // Log file dir added in 2023-11-01 by Chanil Jeon
        }

        //--------------------------------------------------------------------------------------------------
        // Gantry number selection
        this->selectedGantryNumber = parser.get_int("GantryNum", 2);

        // If the user defined number is not 1 or 2, basically set to 2
        if (this->selectedGantryNumber != 1 && this->selectedGantryNumber != 2) this->selectedGantryNumber = 2;

        std::cout << "Selected gantry number : " << this->selectedGantryNumber << std::endl;

        //--------------------------------------------------------------------------------------------------

        // Check if you want to use water phantom geometry
        this->usingPhantomGeo = parser.get_bool("UsingPhantomGeo", false);
        this->twoCentimeterMode = parser.get_bool("twoCentimeterMode", false);
        this->phantomDimX = parser.get_int("PhantomDimX", 200);
        this->phantomDimY = parser.get_int("PhantomDimY", 200);
        this->phantomDimZ = parser.get_int("PhantomDimZ", 200);
        this->phantomUnitX = parser.get_int("PhantomUnitX", 1);
        this->phantomUnitY = parser.get_int("PhantomUnitY", 1);
        this->phantomUnitZ = parser.get_int("PhantomUnitZ", 1);
        this->phantomPositionX = parser.get_int("PhantomPositionX", 1);
        this->phantomPositionY = parser.get_int("PhantomPositionY", 1);
        this->phantomPositionZ = parser.get_int("PhantomPositionZ", 1);

        // ---------------------------------------------------------------------------------------------
        if (!parent_dir.empty()) std::cout << "Reading patient directory.. : Patient directory name --> " << parent_dir << std::endl;

        if (parent_dir.empty()) { throw std::runtime_error("ParentDir is not provided"); }
        if (dicom_dir.empty()) { throw std::runtime_error("DICOM Dir is not provided"); }
        if (logfile_dir.empty()) { throw std::runtime_error("Log Dir is not provided"); }

        // ---------------------------------------------------------------------------------------------
        /// Source parameters
        source_type = parser.get_string("SourceType", "FluenceMap");
        sim_type    = parser.string_to_sim_type(parser.get_string("SimulationType", "perBeam"));
        particles_per_history = parser.get_float("ParticlesPerHistory", -1.0);

        // -------------------------------------------------------------------------------------------
        /// Scorer parameters
        this->scorer_string =
          parser.get_string("Scorer", "EnergyDeposition");   // assumes only one scorer
        scorer_type = parser.string_to_scorer_type(this->scorer_string);

        // -------------------------------------------------------------------------------------------
        //// Set simulation type to per spot for dose dij matrix scoring
        if (this->scorer_type == mqi::DOSE_Dij) { this->sim_type = mqi::PER_SPOT; }
        score_variance          = !parser.get_bool("SupressStd", true);
        score_to_ct_grid        = parser.get_bool("ScoreToCTGrid", true);
        scoring_mask            = parser.get_bool("ScoringMask", false);
        ct_clipping             = false;   //parser.get_bool("CTClipping", false);
        this->body_contour_name = parser.get_string("BodyContourName", "External");
        this->read_structure    = parser.get_bool("ReadStructure", false);

        // If user use phantom geometry, override reading structure false
        if (this->usingPhantomGeo) this->read_structure = false; 

        if (scoring_mask) {
            save_scorer_map = parser.get_bool("SaveMap", true);
            mask_filenames  = parser.get_string_vector("Mask", ",");
            if (mask_filenames.size() ==  0) {
                throw std::runtime_error("Mask filename is missing");
            }
        } else {
            save_scorer_map = false;
            mask_filenames  = {};
        }

        // --------------------------------------------------
        // Masking file
        if (mask_filenames.size() > 0) std::cout << "Reading masking file.. : Masking file count --> " << mask_filenames.size() << std::endl;
        else std::cout << "Reading masking file.. : There is no masking file to read." << std::endl;

        if (save_scorer_map) {
            scorer_map_prefix = parser.get_string("ScorerMapName", "scorer_map");
        } else {
            scorer_map_prefix = "";
        }
        score_to_ct_grid = true;
        if (!score_to_ct_grid) {
            /// TODO: dose grid scoring
            scorer_voxel_size = parser.get_float_vector("ScorerVoxelSize", ",");
            if (scorer_voxel_size.size() == 0) { scorer_voxel_size = { 0, 0, 0 }; }
        }

        // --------------------------------------------------
        /// Output parameters
        output_path   = parser.get_string("OutputDir", "");
        output_format = parser.get_string("OutputFormat", "raw");
        if (strcasecmp(output_format.c_str(), "npz") == 0) {
            this->reshape_output = false;
            this->sparse_output  = true;
        } else {
            this->reshape_output = true;
            this->sparse_output  = false;
        }
        if (output_path.empty()) { throw std::runtime_error("Output directory is not provided."); }
        else
        {
            std::cout << "Setting output.. : Output directory name --> " << output_path << std::endl;
            std::cout << "Setting output.. : Output format --> " << output_format << std::endl;
        }
        overwrite_results = parser.get_bool("OverwriteResults", false);
        if (stat(output_path.c_str(), &info) != 0) {
            mkdir(output_path.c_str(), 0755);
        } else if (!overwrite_results) {
            throw std::runtime_error("Output directory exists.");
        }

        // --------------------------------------------------
        ///Initialize data
        // Reading DICOM CT and RT structure
        this->dcm_ = this->read_dcm_dir();
        if (this->scorer_type == mqi::DOSE || this->scorer_type == mqi::LETd ||
            this->scorer_type == mqi::LETt) {
            this->scorer_capacity = this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z;
        }

        std::string machineName = "";
        std::string referenceMCName = "";

        // Creating treatment machine object from plan information
        tx = new mqi::treatment_session<R>(dcm_.plan_name, machineName, referenceMCName, this->selectedGantryNumber);
        if (sim_type == mqi::PER_BEAM) {
            beam_numbers = parser.get_int_vector("BeamNumbers", ",");
            if (beam_numbers.size() == 0) {
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    //                    printf("%s\n", this->tx->get_beam_name(k).c_str());
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            } else if (beam_numbers.size() == 1 && beam_numbers[0] == 0) {
                beam_numbers.clear();
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            }
        } else if (sim_type == mqi::PER_SPOT) {
            ct_clipping  = false;
            beam_numbers = parser.get_int_vector("BeamNumbers", ",");
            if (beam_numbers.size() == 0) {
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            } else if (beam_numbers.size() == 1 && beam_numbers[0] == 0) {
                beam_numbers.clear();
                for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                    if (this->tx->get_beam_name(k) == "Setup") continue;
                    beam_numbers.push_back(k);
                }
            }
        } else if (sim_type == mqi::PER_PATIENT) {
            beam_numbers.clear();
            for (int k = 1; k < this->tx->get_num_beams() + 1; k++) {
                beam_numbers.push_back(k);
            }
        }
    }

    CUDA_HOST
    ~tps_env() {
        ;
    }

    CUDA_HOST
    virtual void
    print_parameters() {
        printf("================================\n");
        printf("Global parameters\n");
        printf("================================\n");
        printf("Input parameter file: %s\n", input_filename.c_str());
        printf("GPU_ID %d\n", this->gpu_id);
        printf("Random seed %d\n", master_seed);
        printf("The number of total threads %d\n", this->num_total_threads);
        printf("Maximum histories per batch %lu\n", max_histories_per_batch);
        printf("================================\n");
        printf("Setup parameters\n");
        printf("================================\n");
        printf("Patient directory %s\n", parent_dir.c_str());
        printf("DICOM directory %s\n", dicom_dir.c_str());
        printf("Log file directory %s\n", logfile_dir.c_str());
        printf("Scorer type %d\n", this->scorer_type);
        printf("Supress variance %d\n", !score_variance);
        printf("Particles per histories %.1f\n", particles_per_history);
        printf("Source type %s\n", source_type.c_str());
        printf("Simulation type %d\n", sim_type);
        if (sim_type == mqi::PER_BEAM) {
            printf("Beam numbers ");
            for (int i = 0; i < beam_numbers.size(); i++) {
                printf("%d ", beam_numbers[i]);
            }
            printf("\n");
        }
        printf("Machine name %s\n", machine_name.c_str());
        printf("Score CT grid %d\n", score_to_ct_grid);
        printf("Scoring mask %d\n", scoring_mask);
        printf("Save scorer map %d\n", save_scorer_map);
        if (save_scorer_map) { printf("Scorer map save prefix %s\n", scorer_map_prefix.c_str()); }
        printf("Using absolute path %d\n", use_absolute_path);
        printf("Beam prefix %s\n", beam_prefix.c_str());
        printf("================================\n");
        printf("Output parameters\n");
        printf("================================\n");
        printf("Output path %s\n", output_path.c_str());
        printf("Output format %s\n", output_format.c_str());
        printf("Overwrite output %d\n", overwrite_results);

        if (scoring_mask) {
            printf("Mask filenames\n");
            for (int i = 0; i < mask_filenames.size(); i++) {
                printf("%s\n", mask_filenames[i].c_str());
            }
        }
    }

    CUDA_HOST
    virtual struct dicom_t
    read_dcm_dir() 
    {
        // Declare DICOM dataset variable
        dicom_t         dcm;
        gdcm::Directory d;
        //// Need to check the directory is valid and stop process if not
        d.Load(dicom_dir.c_str());
        std::cout << "Reading DICOM directory.. : DICOM directory name --> " << dicom_dir << std::endl;
        const gdcm::Directory::FilenamesType& l1 = d.GetFilenames();
        dcm.nfiles                               = l1.size();
        gdcm::Scanner   s0;

        // Reading modality tag of DICOM
        const gdcm::Tag Modality(0x0008, 0x0060);   // Modality
        s0.AddTag(Modality);
        bool b = s0.Scan(d.GetFilenames());
        if (!b) 
        {
            std::cerr << "Scanner failed" << std::endl;
            throw std::runtime_error("Reading DICOM failed.");
            //            return;
        }

        // ----------------------------------------------------------------------------------------------------------
        // Only get the DICOM files:
        // DICOM RT PLAN : for beam information and geometry
        // DICOM RT Struct : for CT masking ROI
        // DICOM CT : for patient geometry

        dcm.plan_list   = s0.GetAllFilenamesFromTagToValue(Modality, "RTPLAN");
        dcm.struct_list = s0.GetAllFilenamesFromTagToValue(Modality, "RTSTRUCT");

        // If user don't use phantom geometry, load DICOM CT files
        if (!this->usingPhantomGeo)
        {
            dcm.ct          = new mqi::ct<R>(dicom_dir, false);
            dcm.ct->load_data();

            // Get geometry information from CT
            dcm.dim_     = dcm.ct->get_nxyz();
            dcm.org_dim_ = dcm.ct->get_nxyz();
            dcm.dx     = dcm.ct->get_dx();
            dcm.dy     = dcm.ct->get_dy();
            dcm.org_dz = dcm.ct->get_dz();
        }
        else
        {
            dcm.dim_     = { this->phantomDimX, this->phantomDimY, this->phantomDimZ };
            dcm.org_dim_ = { this->phantomDimX, this->phantomDimY, this->phantomDimZ };
            dcm.dx     = this->phantomUnitX;
            dcm.dy     = this->phantomUnitY;
            dcm.org_dz = new float[this->phantomDimZ];
            for (int i = 0; i < this->phantomDimZ; i++) dcm.org_dz[i] = this->phantomUnitZ;
        }

        // Get plan and struct size
        dcm.n_plan   = dcm.plan_list.size();
        dcm.n_struct = dcm.struct_list.size();

        // Error handling for DICOM files
        if (!this->usingPhantomGeo) assert(dcm.n_plan < dcm.nfiles);
        if (dcm.n_plan > 0) 
        {
            if (dcm.n_plan > 1) printf("There are multiple RTPLANs in the path. Selecting the first one\n");
            dcm.plan_name = dcm.plan_list[0];
        } else 
        {
            dcm.plan_name = "";
            throw std::runtime_error("There is no RTPLAN in the path. Exiting.");
        }

        if (dcm.n_struct > 0) 
        {
            if (dcm.n_struct > 1) printf("There are multiple RTSTRUCTs in the path. Selecting the first one\n");
            dcm.struct_name = dcm.struct_list[0];
        } 
        else 
        {
            dcm.struct_name = "";
        }
        //printf("Loading RT Ion plan from %s\n", dcm.plan_name.c_str());
        std::cout << "Reading DICOM directory.. : Loading RT Ion plan data from " << dcm.plan_name << " .." << std::endl;
        
        if (!this->usingPhantomGeo)
        {
            if (dcm.n_plan > dcm.nfiles || dcm.n_dose > dcm.nfiles || dcm.n_struct > dcm.nfiles ||
            dcm.dim_.z > dcm.nfiles) 
            {
                throw std::runtime_error("Something wrong in reading the directory.");
            }
        }
        

        // ------------------------------------------------------------------------------------------------------------
        if (!this->usingPhantomGeo)
        {
            // array of the dcm.ct contains center of voxels
            float xe0  = dcm.ct->get_x()[0] - dcm.dx / 2.0;
            float ye0  = dcm.ct->get_y()[0] - dcm.dy / 2.0;
            float ze0  = dcm.ct->get_z()[0] - dcm.org_dz[0] / 2.0;

            dcm.org_xe = new float[dcm.dim_.x + 1];
            dcm.org_ye = new float[dcm.dim_.y + 1];
            dcm.org_ze = new float[dcm.dim_.z + 1];

            // -------------------------------------------------------------------------------------------------------------
            // Change the voxel center position to edge positions
            for (int i = 0; i < dcm.dim_.x + 1; i++) dcm.org_xe[i] = xe0 + i * dcm.dx;
            for (int i = 0; i < dcm.dim_.y + 1; i++) dcm.org_ye[i] = ye0 + i * dcm.dy;

            for (int i = 0; i < dcm.dim_.z + 1; i++) 
            {
                if (i == 0)
                    dcm.org_ze[i] = ze0;
                else
                    dcm.org_ze[i] =
                      dcm.org_ze[i - 1] +
                      dcm.org_dz[i - 1];   // need to be modified to deal with varying slice thickness
            }

            dcm.image_center.x = (dcm.org_xe[0] + dcm.dx / 2.0 + dcm.org_xe[dcm.dim_.x] - dcm.dx / 2.0) / 2.0;
            dcm.image_center.y = (dcm.org_ye[0] + dcm.dy / 2.0 + dcm.org_ye[dcm.dim_.y] - dcm.dy / 2.0) / 2.0;
            dcm.image_center.z = (dcm.org_ze[0] + dcm.org_dz[0] / 2.0 + dcm.org_ze[dcm.dim_.z] - dcm.org_dz[dcm.dim_.z - 1] / 2.0) / 2.0;
        
            // -------------------------------------------------------------------------------------------------------
            //// TODO: ct clipping check
            // this->ct_clipping = false;
            // if (this->ct_clipping) 
            // {
            //     //// TODO: ct clipping check
            //     //            dcm = this->clip_ct(dcm);
            // } 
            // else 
            // {
               
            // }
            dcm.xe        = dcm.org_xe;
            dcm.ye        = dcm.org_ye;
            dcm.ze        = dcm.org_ze;
            dcm.dz        = dcm.org_dz;
            this->ct_data = new int16_t[dcm.org_dim_.x * dcm.org_dim_.y * dcm.org_dim_.z];
            std::copy(begin(dcm.ct->get_data()), end(dcm.ct->get_data()), this->ct_data);
        }
        else
        {
            // array of the dcm.ct contains center of voxels
            float xe0  = this->phantomPositionX;
            float ye0  = this->phantomPositionY;
            float ze0  = this->phantomPositionZ;
            dcm.org_xe = new float[dcm.dim_.x + 1];
            dcm.org_ye = new float[dcm.dim_.y + 1];
            dcm.org_ze = new float[dcm.dim_.z + 1];

            // -------------------------------------------------------------------------------------------------------------
            // Change the voxel center position to edge positions
            for (int i = 0; i < dcm.dim_.x + 1; i++) dcm.org_xe[i] = xe0 + i * dcm.dx;
            for (int i = 0; i < dcm.dim_.y + 1; i++) dcm.org_ye[i] = ye0 + i * dcm.dy;

            for (int i = 0; i < dcm.dim_.z + 1; i++) 
            {
                if (i == 0)
                    dcm.org_ze[i] = ze0;
                else
                    dcm.org_ze[i] =
                    dcm.org_ze[i - 1] +
                    dcm.org_dz[i - 1];   // need to be modified to deal with varying slice thickness
            }

            dcm.image_center.x = (dcm.org_xe[0] + dcm.dx / 2.0 + dcm.org_xe[dcm.dim_.x] - dcm.dx / 2.0) / 2.0;
            dcm.image_center.y = (dcm.org_ye[0] + dcm.dy / 2.0 + dcm.org_ye[dcm.dim_.y] - dcm.dy / 2.0) / 2.0;
            dcm.image_center.z = (dcm.org_ze[0] + dcm.org_dz[0] / 2.0 + dcm.org_ze[dcm.dim_.z] - dcm.org_dz[dcm.dim_.z - 1] / 2.0) / 2.0;
        
            dcm.xe        = dcm.org_xe;
            dcm.ye        = dcm.org_ye;
            dcm.ze        = dcm.org_ze;
            dcm.dz        = dcm.org_dz;
        }

        // If there is DICOM RT structure, loading informations from it
        if (dcm.n_struct >= 1 && this->read_structure) 
        {
            printf("Loading RT structure from %s\n", dcm.struct_name.c_str());
            gdcm::Reader struct_reader;
            struct_reader.SetFileName(dcm.struct_name.c_str());
            const bool is_valid_rs = struct_reader.Read();
            if (!is_valid_rs) throw std::runtime_error("Invalid RT structure file.");
            mqi::dataset* struct_ds_ = new mqi::dataset(struct_reader.GetFile().GetDataSet(), true);
            auto          roi_seq    = (*struct_ds_)(gdcm::Tag(0x3006, 0x0020));
            printf("Reading ROI information from RT structure.. --> ROI sequence size : %lu\n", roi_seq.size());
            std::vector<std::string> roi_name;
            std::vector<int>         body_ind;

            // Getting ROI information from RT struct
            for (int roi_ind = 0; roi_ind < roi_seq.size(); roi_ind++) {
                roi_seq[roi_ind]->get_values("ROIName", roi_name);
                //std::cout << "ROI name : " << roi_seq[roi_ind] << std::endl;
                if (strcasecmp(roi_name[0].c_str(), this->body_contour_name.c_str()) == 0) {
                    printf("Reading ROI information from RT structure.. --> ROI name : %s\n", roi_name[0].c_str());
                    roi_seq[roi_ind]->get_values("ROINumber", body_ind);
                    //printf("external id %d\n", body_ind[0]);
                    break;
                }
            }
            auto     roi_contour_seq = (*struct_ds_)(gdcm::Tag(0x3006, 0x0039));
            uint8_t* body_contour = new uint8_t[dcm.org_dim_.x * dcm.org_dim_.y * dcm.org_dim_.z];
            std::vector<int>   refer_roi, contour_num;
            std::vector<float> contour_data;
            mqi::vec3<float>*  contour_points;
            int                points_ind;
            //start = std::chrono::high_resolution_clock::now();
            for (int con_ind = 0; con_ind < roi_contour_seq.size(); con_ind++) {
                roi_contour_seq[con_ind]->get_values("ReferencedROINumber", refer_roi);
                if (refer_roi[0] == body_ind[0]) {
                    auto contour_seq = (*roi_contour_seq[con_ind]) (gdcm::Tag(0x3006, 0x0040));
                    for (int contour_ind = 0; contour_ind < contour_seq.size(); contour_ind++) {
                        contour_seq[contour_ind]->get_values("NumberOfContourPoints", contour_num);
                        contour_seq[contour_ind]->get_values("ContourData", contour_data);
                        contour_points = new mqi::vec3<float>[contour_num[0]];
                        for (points_ind = 0; points_ind < contour_num[0]; points_ind++) {
                            contour_points[points_ind].x = contour_data[points_ind * 3];
                            contour_points[points_ind].y = contour_data[points_ind * 3 + 1];
                            contour_points[points_ind].z = contour_data[points_ind * 3 + 2];
                        }
                        fill_contour(body_contour,
                                     contour_points,
                                     contour_num[0],
                                     dcm.dim_,
                                     dcm.xe,
                                     dcm.ye,
                                     dcm.ze,
                                     dcm.dx,
                                     dcm.dy);
                        delete[] contour_points;
                    }
                    break;
                }
            }
            //stop     = std::chrono::high_resolution_clock::now();
            //duration = stop - start;
            //printf("Contour conversion to volume: %f ms\n", duration.count());
            dcm.body_contour = body_contour;
        } 
        else if (this->read_structure) 
        {
            throw std::runtime_error("RT STRCUTURE does not exist");
        }
        return dcm;
    }

    /* Log file reading function, added in 2023-11-02 by Chanil Jeon */
    // Function for log file reading
    CUDA_HOST
    virtual struct logfiles_t
    read_logfile_dir(int beamIndex)
    {
        // Declare log file data variable
        logfiles_t logFileData;

        std::cout << "---------------------------------------------------------------------------" << std::endl;
        std::cout << "Starting process of loading log files.." << std::endl;
        std::cout << "---------------------------------------------------------------------------" << std::endl;

        // Check log file directory is correct
        printf("Reading log file directory.. : Directory name --> %s\n", logfile_dir.c_str());

        std::vector<std::string> fileFolderName;

        /* Get log files data using file system API */
        std::filesystem::path logFileFolderPath(logfile_dir);

        // Get path of field folders
        for (const std::filesystem::directory_entry& entry :
            std::filesystem::directory_iterator(logFileFolderPath)) 
        {
            fileFolderName.push_back(entry.path().string());
        }

        std::sort(fileFolderName.begin(), fileFolderName.end());

        // Read log file data (csv) in each field folder
        for (int i = 0; i < fileFolderName.size(); i++)
        {
            if (i == beamIndex)
            {
                std::filesystem::path p(fileFolderName[i]);

                // Declare log file energy, spot XY, particle count information for each field
                std::vector<logfile_t> fieldLogFileContainer;
                std::vector<float> fieldBeamEnergy;

                std::vector<std::filesystem::path> sortedByName;
                for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(p)) sortedByName.push_back(entry.path());
                std::sort(sortedByName.begin(), sortedByName.end());

                for (auto &filename : sortedByName) 
                {
                    // Check the reading files are csv
                    if (filename.extension() == ".csv")
                    {
                        // Reading each log file data
                        std::fstream fs(filename.string());
                        std::string strBuf;

                        // Get beam energy from file name string
                        std::string logBeamEnergyString = filename.filename().string();
                        logBeamEnergyString = logBeamEnergyString.substr(0, logBeamEnergyString.find('.csv') - 1);
                        logBeamEnergyString = logBeamEnergyString.substr(logBeamEnergyString.find('_') + 1, logBeamEnergyString.find('M') - 2);
                        float layerBeamEnergy = std::stof(logBeamEnergyString);
                        fieldBeamEnergy.push_back(layerBeamEnergy);

                        std::cout << "Reading log file information in the field directory.. : " + filename.string() + ".." << std::endl;
                        
                        // Declare variable of a log file 
                        logfile_t selectedLogFiledInfo;

                        int csvIndex = 0;

                        while (!fs.eof())
                        {
                            if (std::getline(fs, strBuf, ','))
                            {
                                csvIndex += 1; // Add csv index

                                if (csvIndex == 2) selectedLogFiledInfo.posX.push_back(stof(strBuf));
                                else if (csvIndex == 3) selectedLogFiledInfo.posY.push_back(stof(strBuf));
                                else if (csvIndex == 4) 
                                {
                                    selectedLogFiledInfo.muCount.push_back(stoi(strBuf) * this->particles_per_history);
                                    csvIndex = 0;
                                }
                            }
                        }
                        fs.close();
                        fieldLogFileContainer.push_back(selectedLogFiledInfo);
                    }
                }
                logFileData.beamInfo.push_back(fieldLogFileContainer);
                logFileData.beamEnergyInfo.push_back(fieldBeamEnergy);

                std::cout << "Reading log file information for beam " << beamIndex + 1 << " is complete!" << std::endl;
            }
        }
        return logFileData;
    }

    CUDA_HOST
    virtual void
    setup_materials() {
        this->n_materials  = 3;
        this->materials    = new mqi::material_t<R>[this->n_materials];
        this->materials[0] = mqi::air_t<R>();
        this->materials[1] = mqi::h2o_t<R>();
        this->materials[2] = mqi::brass_t<R>();
    }

    CUDA_HOST
    virtual void
    setup_world() {
        this->world = new mqi::node_t<R>;
        ///< By default, let's set +- 40 cm as world volume
        //		const R hl   = 600.0;
        const R hl   = 800.0;
        R       x[2] = { this->dcm_.org_xe[0] - (float) 0.5 * hl,
                   this->dcm_.org_xe[this->dcm_.org_dim_.x] + (float) 0.5 * hl };
        R       y[2] = { this->dcm_.org_ye[0] - (float) 0.5 * hl,
                   this->dcm_.org_ye[this->dcm_.org_dim_.y] + (float) 0.5 * hl };
        R       z[2] = { this->dcm_.org_ze[0] - (float) 0.5 * hl,
                   this->dcm_.org_ze[this->dcm_.org_dim_.z] + (float) 0.5 * hl };

        // When we use custom phantom geometry, add air geometry to child of the world geometry
        // Therefore, this varaiable gives additional size to geometry container
        int beamlineObjectCorrection = 0;
        int worldChildCorrection = 0;
        //if (this->usingPhantomGeo) phantomGeoCorrection = 1;
        if (this->usingPhantomGeo)
        {
            if (this->twoCentimeterMode)
            {
                this->phantomDimX = 400;
                this->phantomDimY = 1;
                this->phantomDimZ = 400;
                this->phantomUnitX = 1.f;
                this->phantomUnitY = 2.f;
                this->phantomUnitZ = 1.f;
                this->phantomPositionX = -200.f;
                this->phantomPositionY = -1.f;
                this->phantomPositionZ = -200.f;
                beamlineObjectCorrection = 1;
                worldChildCorrection = 3;
            }
            else
            {
                beamlineObjectCorrection = 1;
                worldChildCorrection = 1;
            }
        }

        // Because the program don't calculate anything in world geometry, set rho_mass to 0
        this->world->geo = new mqi::grid3d<density_t, R>(x, 2, y, 2, z, 2);           
        this->world->geo->fill_data(0.0); // There is no meaning that add some density to world geometry
        this->world->n_scorers                          = 0;
        this->world->scorers                            = nullptr;
        mqi::beamline<R>            beamline            = this->tx->get_beamline(bnb);
        std::vector<mqi::geometry*> beamline_geometries = beamline.get_geometries();
        this->world->n_children                         = beamline_geometries.size() + 1 + worldChildCorrection; // Original is + 1. Additional + 1 for air box
        this->world->children                           = new node_t<R>*[this->world->n_children]; // Beam line geometry + airbox

        ///< Create beamline objects
        mqi::coordinate_transform<R> p_coord = this->tx->get_coordinate(bnb);
        p_coord.angles[3]                    = 90.0;   //iec2dicom angle
        node_t<R>** beamline_objects         = new node_t<R>*[beamline_geometries.size() + beamlineObjectCorrection];

        if (this->usingPhantomGeo)
        {
            p_coord.translation.x = 0.f;
            p_coord.translation.y = 0.f;
            p_coord.translation.z = 0.f;

            p_coord.angles[0] = 0.f;
            p_coord.angles[1] = 0.f;
            p_coord.angles[2] = 0.f;
            p_coord.angles[3] = 0.f;
        }

        if (this->usingPhantomGeo)
        {
            // Get snout position from ion control point sequence
            const mqi::dataset* beamDataset = this->tx->get_beam_dataset(bnb); // Get beam dataset
            const mqi::dataset* icps = (*beamDataset)("IonControlPointSequence")[0]; // Get isocenter position
            std::vector<float>  snoutPos;
            icps->get_values("SnoutPosition", snoutPos); // Get snout position

            // Creating airbox between beam start and right before the phantom surface
            // Set air box to end index of beamline object, becuase the simulation calculates sequentially
            node_t<R>* airBox  = new node_t<R>;
            beamline_objects[beamline_geometries.size()] = airBox;
            this->world->children[beamline_geometries.size()] = beamline_objects[beamline_geometries.size()];

            double airEndPos{};
            if (this->twoCentimeterMode) airEndPos = 20.f;
            else airEndPos = this->phantomPositionZ + this->phantomDimZ * phantomUnitZ;

            airBox->geo = new grid3d<density_t, R>(-200,
                                            200,
                                            2,
                                            -200,
                                            200,
                                            2,
                                            airEndPos,
                                            snoutPos[0], // Snout position - gap for calculation
                                            2,//int((snoutPos[0] - (airEndPos)) / 2) + 1,
                                            p_coord.rotation);
            airBox->geo->fill_data(mqi::air_t<R>().rho_mass); // Fill volume with air
            
            // Air box has no scorer and children
            airBox->n_scorers = 0;
            airBox->n_children = 0;
            airBox->scorers = nullptr;
            airBox->children = nullptr;
        }

        for (int i = 0; i < beamline_geometries.size(); i++) {
            beamline_objects[i] = new node_t<R>;
            if (beamline_geometries[i]->geotype == mqi::RANGESHIFTER) {
                printf("Adding rangershifter geometry..\n");
                beamline_objects[i] = this->create_rangeshifter(
                  dynamic_cast<mqi::rangeshifter*>(beamline_geometries[i]), p_coord);
            } else if (beamline_geometries[i]->geotype == mqi::BLOCK) {
                /// TODO: defining voxelized aperture
                //                beamline_objects[i+1] = this->create_voxelixed_aperture(
                //                  dynamic_cast<mqi::aperture*>(beamline_geometries[i+1]));
            }
            /// TODO: dealing with aperture
            beamline_objects[i]->n_scorers = 0;
            this->world->children[i] = beamline_objects[i];
        }

        ///< create a child
        std::cout << "Creating child in world geometry.. : Phantom size -->" << std::endl;
        if (this->twoCentimeterMode) std::cout << "(x, y, z) -> (400, 400, 2)" << std::endl;
        else std::cout << "(x, y, z) -> (" << this->dcm_.dim_.x << ", " << this->dcm_.dim_.y << ", " << this->dcm_.dim_.z << ")" << std::endl;
        node_t<R>* frontPhantom = new node_t<R>;
        node_t<R>* backPhantom = new node_t<R>;
        node_t<R>* phantom = new node_t<R>;
        
        // 1. If user uses CT geometry
        if (!this->usingPhantomGeo)
        {
            this->world->children[beamline_geometries.size()] = phantom;
            //mqi::material_id* mids = new mqi::material_id[dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z];
            phantom->geo           = new grid3d<density_t, R>(this->dcm_.xe,
                                                    this->dcm_.dim_.x + 1,
                                                    this->dcm_.ye,
                                                    this->dcm_.dim_.y + 1,
                                                    this->dcm_.ze,
                                                    this->dcm_.dim_.z + 1);
            density_t* rho_mass = new density_t[dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z];
            std::cout << "Creating material information for grid.." << std::endl;
            for (int i = 0; i < dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z; i++) 
            {
                rho_mass[i] = this->tx->material_.hu_to_density(this->ct_data[i]);
            }
            phantom->geo->set_data(rho_mass);   //// Material conversion function required
        }
        else // 2. If user uses phantom geometry
        {
            mqi::coordinate_transform<R> transformPhantom = this->tx->get_coordinate(bnb);
            transformPhantom.translation.x = 0.f;
            transformPhantom.translation.y = 0.f;
            transformPhantom.translation.z = 0.f;

            transformPhantom.angles[0] = 0.f;
            transformPhantom.angles[1] = 0.f;
            transformPhantom.angles[2] = 0.f;
            transformPhantom.angles[3] = 0.f;

            if (this->twoCentimeterMode)
            {
                // Front phantom
                this->world->children[beamline_geometries.size() + 1] = frontPhantom;
                frontPhantom->geo           = new grid3d<density_t, R>(-200,
                                                        200,
                                                        401,
                                                        -200,
                                                        200,
                                                        401,
                                                        1,
                                                        20,
                                                        2,
                                                        transformPhantom.rotation);

                density_t* rho_mass_parent1 = new density_t[400 * 400 * 1];
                for (int i = 0; i < 400 * 400 * 1; i++) 
                {
                    rho_mass_parent1[i] = mqi::h2o_t<R>().rho_mass; // Water
                }
                frontPhantom->geo->set_data(rho_mass_parent1);

                this->world->children[beamline_geometries.size() + 2] = phantom;
                phantom->geo           = new grid3d<density_t, R>(-200,
                                                        200,
                                                        401,
                                                        -200,
                                                        200,
                                                        401,
                                                        -1,
                                                        1,
                                                        2,
                                                        transformPhantom.rotation);

                density_t* rho_mass = new density_t[400 * 400 * 1];
                for (int i = 0; i < 400 * 400 * 1; i++) 
                {
                    rho_mass[i] = mqi::h2o_t<R>().rho_mass; // Water
                }
                phantom->geo->set_data(rho_mass);

                 // Back phantom
                this->world->children[beamline_geometries.size() + 3] = backPhantom;
                backPhantom->geo           = new grid3d<density_t, R>(-200,
                                                        200,
                                                        401,
                                                        -200,
                                                        200,
                                                        401,
                                                        -380,
                                                        -1,
                                                        2,
                                                        transformPhantom.rotation);

                density_t* rho_mass_parent2 = new density_t[400 * 400 * 189];
                for (int i = 0; i < 400 * 400 * 1; i++) 
                {
                    rho_mass_parent2[i] = mqi::h2o_t<R>().rho_mass; // Water
                }
                backPhantom->geo->set_data(rho_mass_parent2);
            }
            else
            {
                this->world->children[beamline_geometries.size() + 1] = phantom;
                phantom->geo           = new grid3d<density_t, R>(this->dcm_.xe,
                                                        this->dcm_.dim_.x + 1,
                                                        this->dcm_.ye,
                                                        this->dcm_.dim_.y + 1,
                                                        this->dcm_.ze,
                                                        this->dcm_.dim_.z + 1,
                                                        transformPhantom.rotation);

                density_t* rho_mass = new density_t[dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z];
                for (int i = 0; i < dcm_.dim_.x * dcm_.dim_.y * dcm_.dim_.z; i++) 
                {
                    rho_mass[i] = mqi::h2o_t<R>().rho_mass; // Water
                }
                phantom->geo->set_data(rho_mass);
            }
        }

        // Mask reading
        mqi::mask_reader mask_reader0(this->dcm_.dim_);
        roi_t*           roi_tmp;
        if (scoring_mask) {
            mask_reader0.mask_filenames = mask_filenames;
            mask_reader0.read_mask_files();
            roi_tmp = mask_reader0.mask_to_roi();
        } else if (this->read_structure) {
            mask_reader0.set_mask(this->dcm_.body_contour);
            roi_tmp = mask_reader0.mask_to_roi();
        } else {
            roi_tmp =
              new roi_t(mqi::DIRECT, this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z);
        }
        if (this->scorer_type == mqi::LETd || this->scorer_type == mqi::LETt) {
            phantom->n_scorers = 2;   // need two scorers for LET scoring
        } else {
            phantom->n_scorers = 1;
        }
        phantom->n_scorers = 1;

        phantom->scorers = new scorer<R>*[phantom->n_scorers];
        fp_compute_hit<R> fp0;

#if defined(__CUDACC__)
        cudaMemcpyFromSymbol(&fp0, mqi::Dw_pointer, sizeof(fp_compute_hit<R>));
#else
        fp0             = mqi::dose_to_water;
#endif
        phantom->scorers[0] =
          new mqi::scorer<R>(this->scorer_string.c_str(),
                             this->dcm_.dim_.x * this->dcm_.dim_.y * this->dcm_.dim_.z,
                             fp0);

        mqi::key_value* deposit0 = new mqi::key_value[phantom->scorers[0]->max_capacity_];

        std::memset(deposit0, 0xff, sizeof(mqi::key_value) * phantom->scorers[0]->max_capacity_);

        init_table(deposit0, phantom->scorers[0]->max_capacity_);

        phantom->scorers[0]->data_           = deposit0;
        phantom->scorers[0]->score_variance_ = this->score_variance;
        phantom->scorers[0]->roi_            = roi_tmp;

        if (this->score_variance) {
            printf("Score_variance\n");
            mqi::key_value** count        = new mqi::key_value*[phantom->n_scorers];
            mqi::key_value** vox_mean     = new mqi::key_value*[phantom->n_scorers];
            mqi::key_value** vox_variance = new mqi::key_value*[phantom->n_scorers];
            for (int s_ind; s_ind < phantom->n_scorers; s_ind++) {
                count[s_ind]        = new mqi::key_value[phantom->scorers[s_ind]->max_capacity_];
                vox_mean[s_ind]     = new mqi::key_value[phantom->scorers[s_ind]->max_capacity_];
                vox_variance[s_ind] = new mqi::key_value[phantom->scorers[s_ind]->max_capacity_];
                phantom->scorers[s_ind]->count_    = count[s_ind];
                phantom->scorers[s_ind]->mean_     = vox_mean[s_ind];
                phantom->scorers[s_ind]->variance_ = vox_variance[s_ind];
            }
        }

        mc::mc_score_variance = this->score_variance;
    }

    // Beam source loading code
    CUDA_HOST
    virtual void
    setup_beamsource() {
        uint16_t                 num_beams  = this->tx->get_num_beams();
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::cout << "Creating beam source object.. : Detected beam count --> " << num_beams << std::endl;
        std::cout << "Creating beam source object.. : Currently selected beam --> " << bnb << " (" << beam_names[bnb - 1] << ")" << std::endl;

        const mqi::dataset* mqi_ds            = this->tx->get_beam_dataset(bnb);
        const mqi::dataset* ion_beam_sequence = (*mqi_ds)("IonControlPointSequence")[0];
        std::vector<float>  temp_sid;
        ion_beam_sequence->get_values("SnoutPosition", temp_sid);
        this->sid = temp_sid[0];
        std::cout << "Creating beam source object.. : Detected snout position --> " << this->sid << " mm" << std::endl;
        mqi::coordinate_transform<R> p_coord = this->tx->get_coordinate(bnb);
        p_coord.angles[3]                    = 90.0;   //iec2dicom angle

        if (this->usingPhantomGeo)
        {
            p_coord.translation.x = 0.f;
            p_coord.translation.y = 0.f;
            p_coord.translation.z = 0.f;

            p_coord.angles[0] = 0.f;
            p_coord.angles[1] = 0.f;
            p_coord.angles[2] = 0.f;
            p_coord.angles[3] = 0.f;

        }

        std::cout << "Creating coordinate transform for beam source complete.. : p_coord angle information -- >" << std::endl;
        std::cout << "(" << p_coord.angles[0] << ", " << p_coord.angles[1] << ", " << p_coord.angles[2] << ", " << p_coord.angles[3] << ")" << std::endl; 
        p_coord.translation = p_coord.translation;   // - this->dcm_.image_center;
        mqi::coordinate_transform<R> p_final(p_coord.angles, p_coord.translation);

        // Beam line translation when range shifter used
        bool rangeShifterUsed{ false };
        std::vector<int> rangeShifterCount;
        this->tx->get_beam_dataset(bnb)->get_values("NumberOfRangeShifters", rangeShifterCount);
        if (rangeShifterCount[0] >= 1) rangeShifterUsed = true;

        std::cout << "Creating coordinate transform for beam source complete.. : Translation information -- > " << std::endl;
        p_final.translation.dump();
        std::cout << "Creating coordinate transform for beam source complete.. : Rotation information -- > " << std::endl;
        p_final.rotation.dump();

        // ****************************************************************************
        // Reading log file
        //std::cout << "Loading beam number index : " << bnb-1 << std::endl;
        logfiles_t logFileData = read_logfile_dir(bnb-1);

        // Get beam source information using RT ION PLAN and log file
        // Modified in 2023 by Chanil Jeon
        // bnb => Beam ID
        // p_final => Coordinate transform by DICOM RT ION PLAN
        // sid => Source to isocenter distance
        // Read all spot information at once
        this->beamsource = this->tx->get_beamsource(logFileData, p_final, sid, rangeShifterUsed);
        // ****************************************************************************

        std::cout << "Total beamlets (Spots) : " << this->beamsource.total_beamlets() << std::endl;
        std::cout << "Total histories (Particles) : " << this->beamsource.total_histories() << std::endl;

        std::cout << "Creating beam source complete!" << std::endl;
    }

    CUDA_HOST
    void
    initialize_and_run() {
        for (int beam_queue = 0; beam_queue < beam_numbers.size(); beam_queue++) {
            this->bnb = beam_numbers[beam_queue];
            this->master_seed += beam_queue * 10000;
            this->beam_rng.seed(this->master_seed);
            //printf("bnb %d seed %d\n", this->bnb, this->master_seed);
            this->initialize();
            this->run();
            this->finalize();
            if (this->reshape_output) {
                this->save_reshaped_files();
            } else if (this->sparse_output) {
                this->save_sparse_file();
            }
        }
    }
    CUDA_HOST
    virtual void
    run() {
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        std::cout << "Starting process of Monte Carlo simulation.." << std::endl;
        std::cout << "---------------------------------------------------------------------------" << std::endl;
        size_t free, total;
        //printf("Selected scorer type ; %d\n", scorer_type, sim_type);
        if (this->sim_type == mqi::PER_BEAM) {
#if defined(__CUDACC__)
            cudaMemGetInfo(&free, &total);
            //printf("Geometry occupies %f GB\n", (total - free) / (1024.0 * 1024.0 * 1024.0));
            std::cout << "Starting simulation per field.. : Geometry allocated memory --> " << (total - free) / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
#endif
            run_by_beam();
        } else if (this->sim_type == mqi::PER_SPOT) {
#if defined(__CUDACC__)
            cudaMemGetInfo(&free, &total);
            printf("Geometry occupies %f GB\n", (total - free) / (1024.0 * 1024.0 * 1024.0));
#endif
            run_by_spot();
        }
    }   // run

    CUDA_HOST
    virtual void
    run_simulation(size_t    histories_per_batch,
                   size_t    histories_in_batch,
                   uint32_t* tracked_particles,
                   uint32_t* scorer_offset_vector = nullptr) {
        /// histories_per_batch and histories_in_batch are kine of redundant.
        /// the histories_per_batch may not required if copying memory work correctly with histories_in_batch
        //auto start = std::chrono::high_resolution_clock::now();
        // TODO: divide vertices into sub-vertices if it is too large
        uint32_t                                  n_threads            = 0;
        uint32_t                                  n_blocks             = 0;
        uint32_t                                  particles_per_thread = 0;
        mqi::thrd_t*                              worker_threads;
        //auto                                      stop = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> duration = stop - start;
        //printf("Initialization for run done %f ms\n", duration.count());
#if defined(__CUDACC__)
        //start     = std::chrono::high_resolution_clock::now();
        n_threads = thread_limit;
        if (histories_in_batch < n_threads) {
            n_threads            = histories_in_batch;
            n_blocks             = 1;
            particles_per_thread = histories_in_batch / n_threads;
        } else if (this->num_total_threads > 0) {
            if (this->num_total_threads <= thread_limit) {
                n_threads = this->num_total_threads;
                n_blocks  = 1;
            } else {
                n_threads = thread_limit;
                n_blocks  = (int) std::ceil(this->num_total_threads * 1.0 / n_threads);
                if (n_blocks > block_limit) { n_blocks = n_blocks; }
            }
        } else if (this->num_total_threads < 0) {
            n_threads = n_threads;
            n_blocks  = (int) std::ceil(histories_in_batch * 1.0 / n_threads);
            if (n_blocks > block_limit)   //maybe larger?
                n_blocks = block_limit;
            particles_per_thread =
              (int) std::ceil(histories_in_batch * 1.0 / (n_threads * n_blocks));
            if (histories_in_batch % (n_threads * n_blocks * particles_per_thread) > 0)
                n_blocks += 1;   // increase block size if there is any remainder
            // assert(n_blocks*n_threads>histories_in_batch && (n_blocks-1)*n_threads<histories_in_batch);
        }
        printf("Printing simulation specification.. : Block size --> %d, Thread size --> %d\n", n_blocks, n_threads);
        mc::upload_vertices(this->vertices, mc::mc_vertices, 0, histories_per_batch);
        cudaDeviceSynchronize();
        check_cuda_last_error("(upload vertices)");
        uint32_t* d_scorer_offset_vector;
        if (scorer_offset_vector) {
            printf("Printing simulation specification.. : Histories per batch --> %d\n", histories_per_batch);
            mc::upload_scorer_offset_vector(
              scorer_offset_vector, d_scorer_offset_vector, histories_per_batch);
            cudaDeviceSynchronize();
        } else {
            d_scorer_offset_vector = nullptr;
        }
        uint32_t* d_tracked_particles;
        gpu_err_chk(cudaMalloc(&worker_threads, n_blocks * n_threads * sizeof(mqi::thrd_t)));
        //start = std::chrono::high_resolution_clock::now();
        //printf("master seed %d\n", this->master_seed);
        initialize_threads<<<n_blocks, n_threads>>>(
          worker_threads, n_blocks * n_threads, this->master_seed);
        cudaDeviceSynchronize();
        check_cuda_last_error("(initialize threads)");
        gpu_err_chk(cudaMalloc(&d_tracked_particles, sizeof(tracked_particles[0])));
        gpu_err_chk(cudaMemcpy(d_tracked_particles,
                               tracked_particles,
                               sizeof(tracked_particles[0]),
                               cudaMemcpyHostToDevice));
        printf("Starting transportation call.. \n");
        printf("Printing simulation specification.. : Histories per batch --> %d\n", histories_per_batch);
        mc::transport_particles_patient<R><<<n_blocks, n_threads>>>(
          worker_threads, mc::mc_world, mc::mc_vertices, histories_in_batch, d_tracked_particles);
        cudaDeviceSynchronize();
        check_cuda_last_error("(transport particle table)");

        printf("Transportation call ended!\n");
        gpu_err_chk(cudaMemcpy(tracked_particles,
                               d_tracked_particles,
                               sizeof(tracked_particles[0]),
                               cudaMemcpyDeviceToHost));
        gpu_err_chk(cudaFree(d_tracked_particles));
        gpu_err_chk(cudaFree(worker_threads));
        gpu_err_chk(cudaFree(mc::mc_vertices));
#else
        n_threads       = 1;
        mc::mc_vertices = this->vertices;
        mc::mc_world    = this->world;
        /// TODO: number of threads for multithreading implementation
        worker_threads = new mqi::thrd_t[n_threads];
        initialize_threads(worker_threads, n_threads, this->master_seed);
        printf("Thread initialization complete!\n");
        mc::transport_particles_patient<R>(
          worker_threads, mc::mc_world, mc::mc_vertices, histories_in_batch, tracked_particles);
#endif
    }   //run_simulation

    CUDA_HOST
    void
    read_vertices_spot(size_t                                      history_start,
                       size_t                                      history_end,
                       std::tuple<mqi::beamlet<R>, size_t, size_t> bl,
                       mqi::vertex_t<R>*                           vertices,
                       uint32_t*                                   score_offset_vector,
                       int                                         spot_ind,
                       size_t                                      histories_per_batch) {
        for (int history_ind = history_start; history_ind < history_end; history_ind++) {
            vertices[history_ind] = std::get<0>(bl)(&this->beam_rng);

            score_offset_vector[history_ind] =
              spot_ind * this->scorer_size;   // Store beamlet index for each history
            assert(history_ind < histories_per_batch);
        }
    }
    CUDA_HOST
    virtual void
    run_by_beam(mqi::node_t<R>* world = mc::mc_world) {
        //// Beam simulation
        /// TODO: faster implementation

        size_t                                                      h0 = 0;
        size_t    h1                = this->beamsource.total_histories();
        uint32_t  num_vertices      = h1 - h0;
        uint32_t* tracked_particles = new uint32_t[1];
        tracked_particles[0]        = 0;

        int    num_batches;
        size_t histories_per_batch = 0, cum_vertices = 0;
        size_t current_vertex = 0;
        if (this->max_histories_per_batch <= 0) {
            num_batches         = 1;
            histories_per_batch = (h1 - h0);   // upload all vertices at once
            std::cout << "Uploading particles with no batch.. : Particle count --> " << histories_per_batch << std::endl;
        } else {
            num_batches = (int) mqi::mqi_ceil((h1 - h0) * 1.0 / this->max_histories_per_batch);
            histories_per_batch = this->max_histories_per_batch;
            std::cout << "Uploading particles with batch.. : Particle count --> " << histories_per_batch << " with " << num_batches << " batches" << std::endl;
        }
        for (int batch = 0; batch < num_batches; batch++) 
        {
            this->vertices = new mqi::vertex_t<R>[histories_per_batch];
            printf("Generating particles for (%d of %d batches) in CPU ..\n", batch + 1, num_batches);
            for (current_vertex = 0; current_vertex < histories_per_batch; current_vertex++) 
            {
                if (cum_vertices + current_vertex >= (h1 - h0)) { break; }
                auto bl = this->beamsource(cum_vertices + current_vertex);
                this->vertices[current_vertex] = bl(&this->beam_rng);   // copy histories to vertices
            }

            std::cout << "Particle generation complete!" << std::endl;
            cum_vertices += current_vertex;
            printf("Transporting particles...\n");
            run_simulation(histories_per_batch, current_vertex, tracked_particles);
            std::cout << "Particle transportation complete!" << std::endl;
            delete[] this->vertices;
            if (tracked_particles[0] == h1) { break; }
        }

    }   //run_by_beam

    // Change RT file based beam generation to log file based generation
    // 2023-11-01

    CUDA_HOST
    virtual void
    run_by_spot(mqi::node_t<R>* world = mc::mc_world) {
        //// Spot by spot simulation
        /// TODO: faster implementation
        printf("Run by spot\n");
        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        std::chrono::duration<double, std::milli>                   duration;
        size_t                                                      h0 = 0;
        size_t h1                   = this->beamsource.total_histories();
        this->num_spots             = this->beamsource.total_beamlets();
        size_t    max_histories     = 0;
        uint32_t* tracked_particles = new uint32_t[1];
        tracked_particles[0]        = 0;
        printf("num spots %d\n", this->num_spots);
        size_t    total_history   = 0;
        uint32_t* spot_boundaries = new uint32_t[this->num_spots];
        size_t    num_histories, cum_histories = 0;
        size_t    num_batches, history_ind, loop_end;
        size_t    cum_vertices = 0, spot_start = 0, batch = 0, spot_ind = 0;
        size_t    current_history     = 0;   // Number of histories from current spot id
        size_t    current_vertex      = 0;   // Number of vertex in current batch
        size_t    histories_per_batch = 0;

        if (this->max_histories_per_batch == 0) {
            num_batches         = 1;
            histories_per_batch = (h1 - h0);   // upload all vertices at onc
            printf("Upload %lu histories\n", histories_per_batch);
        } else {
            num_batches         = (int) mqi::mqi_ceil(h1 * 1.0 / this->max_histories_per_batch);
            histories_per_batch = this->max_histories_per_batch;
            printf("Upload %lu histories per batch, %lu batches expected\n",
                   histories_per_batch,
                   num_batches);
        }

        mqi::vertex_t<R>* vertices_test;
        //        printf("histories per batch %d\n",histories_per_batch);
        while (spot_ind < this->num_spots) {
            this->vertices                = new mqi::vertex_t<R>[histories_per_batch];
            vertices_test                 = new mqi::vertex_t<R>[histories_per_batch];
            uint32_t* score_offset_vector = new uint32_t[histories_per_batch];
            //            printf("num batches %d batch %d spot start %d\n",num_batches,batch, spot_start);
            start = std::chrono::high_resolution_clock::now();
            printf("Generating particles..\n");
            for (spot_ind = spot_start; spot_ind < this->num_spots; spot_ind++) {
                auto bl       = this->beamsource[spot_ind];
                num_histories = std::get<1>(bl);
                if (num_histories - current_history < histories_per_batch - current_vertex) {
                    loop_end = num_histories - current_history + current_vertex;
                } else {
                    loop_end = histories_per_batch;
                }
                /// The multithreading gives small performance gain if we need run it for each spot
                ///20 seconds ->  18 seconds
                read_vertices_spot(current_vertex,
                                   loop_end,
                                   bl,
                                   this->vertices,
                                   score_offset_vector,
                                   spot_ind,
                                   histories_per_batch);

                assert(loop_end > current_vertex);
                current_history += loop_end - current_vertex;
                cum_vertices += loop_end - current_vertex;
                current_vertex += loop_end - current_vertex;

                if (current_vertex == histories_per_batch && current_history < num_histories) {
                    /// The spot have remaining particles to simulate
                    spot_start = spot_ind;
                    break;
                } else if (current_vertex == histories_per_batch &&
                           current_history == num_histories) {
                    /// The spot have no remaining particles to simulate
                    spot_start = spot_ind + 1;
                    break;
                } else {
                    current_history = 0;
                }
            }
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;

            //printf("beam generation %f ms\n", duration.count());
            /// Transport particles
            printf("Transporting particles..\n");
            start = std::chrono::high_resolution_clock::now();
            run_simulation(
              histories_per_batch, current_vertex, tracked_particles, score_offset_vector);
            stop     = std::chrono::high_resolution_clock::now();
            duration = stop - start;
            printf("run simulation %f ms\n", duration.count());
            current_vertex = 0;
            delete[] this->vertices;
            delete[] score_offset_vector;
            batch += 1;
            if (tracked_particles[0] == h1) { break; }
        }
        printf("spot ind %lu num_spots %d cum vertices %lu total histories %lu\n",
               spot_ind,
               this->num_spots,
               cum_vertices,
               h1);
        printf("Number of particles tracked %d\n", tracked_particles[0]);

    }   // run_by_spot

    virtual mqi::node_t<R>*
    create_rangeshifter(mqi::rangeshifter* geometry, mqi::coordinate_transform<R> p_coord) {
        mqi::node_t<R>* rangeshifter = new mqi::node_t<R>;
        rangeshifter->n_scorers      = 0;
        rangeshifter->n_children     = 0;
        rangeshifter->scorers        = nullptr;
        rangeshifter->children       = nullptr;

        printf("Printing rangeshifter specification.. : p_coord angle of rangeshifter --> \n(%f %f %f %f)\n",
               p_coord.angles[0],
               p_coord.angles[1],
               p_coord.angles[2],
               p_coord.angles[3]);

        printf("Printing rangeshifter specification.. : p_coord position of rangeshifter --> \n(%f %f %f)\n",
               p_coord.translation.x,
               p_coord.translation.y,
               p_coord.translation.z);

        mqi::coordinate_transform<R> p_final(p_coord.angles, p_coord.translation);

        std::cout << "Printing rangeshifter specification.. : Position -->" << std::endl;
        geometry->pos.dump();
        std::cout << "Printing rangeshifter specification.. : Volume -->" << std::endl;
        geometry->volume.dump();
        std::cout << "Creating rangeshifter grid.." << std::endl;
        rangeshifter->geo = new grid3d<mqi::density_t, R>(geometry->pos.x - geometry->volume.x / 2,
                                                          geometry->pos.x + geometry->volume.x / 2,
                                                          2,
                                                          geometry->pos.y - geometry->volume.y / 2,
                                                          geometry->pos.y + geometry->volume.y / 2,
                                                          2,
                                                          geometry->pos.z - geometry->volume.z / 2,
                                                          geometry->pos.z + geometry->volume.z / 2,
                                                          2,
                                                          p_final.rotation);

        std::cout << "Filling rangeshifter grid with specific density.." << std::endl;
        /// TODO: Check material and density of rangeshifter
        // For SMC, 4 cm of solid water phantom is range shifter.
        // 39.37 mm is water-equivalent length
        rangeshifter->geo->fill_data(mqi::h2o_t<R>().rho_mass); // Water
        rangeshifter->geo->translation_vector = p_final.translation;
        //std::cout << "Printing rangeshifter specification.. : Rotation for coordinate transform -->" << std::endl;
        //p_final.rotation.dump();
        std::cout << "Printing rangeshifter specification.. : Rotation for rangeshifter geometry -->" << std::endl;
        rangeshifter->geo->rotation_matrix_fwd.dump();
        std::cout << "Creating rangeshifter complete!" << std::endl;

        return rangeshifter;
    }
    CUDA_HOST_DEVICE
    bool
    sol1_1(mqi::vec2<float> pos, mqi::vec3<float>*& contour_points, int num_points) {
        mqi::vec3<float> pos0 = contour_points[0];
        mqi::vec3<float> pos1;
        float            min_y, max_y, max_x, intersect_x;
        int              count = 0;
        int              i, j, c = 0;
        for (i = 0, j = num_points - 1; i < num_points; j = i++) {
            pos0 = contour_points[i];
            pos1 = contour_points[j];
            if ((((pos0.y <= pos.y) && (pos.y < pos1.y)) ||
                 ((pos1.y <= pos.y) && (pos.y < pos0.y))) &&
                (pos.x < (pos1.x - pos0.x) * (pos.y - pos0.y) / (pos1.y - pos0.y) + pos0.x)) {
                c = !c;
            }
        }
        return c;
    }
    CUDA_HOST_DEVICE
    void
    fill_contour(uint8_t*           volume_contour,
                 mqi::vec3<float>*& contour_points,
                 int                num_contour_points,
                 mqi::vec3<ijk_t>   dim,
                 const R*           x_pix,
                 const R*           y_pix,
                 const R*           z_pix,
                 float              dx,
                 float              dy) {
        int x_ind = 0, y_ind = 0, z_ind = -1;
        for (int i = 0; i < dim.z - 1; i++) {
            if (contour_points[0].z > z_pix[i] && contour_points[0].z < z_pix[i + 1]) {
                z_ind = i;
                break;
            }
        }
        mqi::vec2<float> pos;
        uint32_t         idx = 0;
        float            x_pos, y_pos;
        if (z_ind >= 0) {
            for (x_ind = 0; x_ind < dim.x - 1; x_ind++) {
                for (y_ind = 0; y_ind < dim.y - 1; y_ind++) {
                    pos.x = x_pix[x_ind] + dx * 0.5;
                    pos.y = y_pix[y_ind] + dy * 0.5;
                    idx   = z_ind * dim.x * dim.y + y_ind * dim.x + x_ind;
                    assert(idx < dim.x * dim.y * dim.z);
                    if (sol1_1(pos, contour_points, num_contour_points)) {
                        volume_contour[idx] = 1;
                    } else {
                        //                        volume_contour[idx] = 0;
                    }
                }
            }
        }
    }

    CUDA_HOST
    double*
    reshape_data(int c_ind, int s_ind, mqi::vec3<ijk_t> dim) {
        //        R* reshaped_data = new R[dim.x * dim.y * dim.z];
        double* reshaped_data = new double[dim.x * dim.y * dim.z];
        //        std::memset(reshaped_data, 0, sizeof(R) * dim.x * dim.y * dim.z);
        int ind_x = 0, ind_y = 0, ind_z = 0, lin = 0;
        for (int i = 0; i < dim.x * dim.y * dim.z; i++) {
            reshaped_data[i] = 0;
        }
        //printf("max capacity %d\n", this->world->children[c_ind]->scorers[s_ind]->max_capacity_);
        for (int ind = 0; ind < this->world->children[c_ind]->scorers[s_ind]->max_capacity_;
             ind++) {
            if (this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1 != mqi::empty_pair &&
                this->world->children[c_ind]->scorers[s_ind]->data_[ind].key2 != mqi::empty_pair) {
                reshaped_data[this->world->children[c_ind]->scorers[s_ind]->data_[ind].key1] +=
                  this->world->children[c_ind]->scorers[s_ind]->data_[ind].value;
            }
        }
        return reshaped_data;
    }

    CUDA_HOST
    void
    save_reshaped_files() {
        uint32_t                 vol_size;
        mqi::vec3<ijk_t>         dim;
        double*                  reshaped_data;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim           = this->world->children[c_ind]->geo->get_nxyz();
                vol_size      = dim.x * dim.y * dim.z;
                reshaped_data = this->reshape_data(c_ind, s_ind, dim);
                if (!this->output_format.compare("mhd")) {
                    mqi::io::save_to_mhd<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            this->particles_per_history,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else if (!this->output_format.compare("mha")) {
                    mqi::io::save_to_mha<R>(this->world->children[c_ind],
                                            reshaped_data,
                                            this->particles_per_history,
                                            this->output_path,
                                            filename,
                                            vol_size);
                } else {
                    mqi::io::save_to_bin<double>(reshaped_data,
                                                 this->particles_per_history,
                                                 this->output_path,
                                                 filename,
                                                 vol_size);
                }

                delete[] reshaped_data;
            }
        }
    }

    CUDA_HOST
    virtual void
    save_sparse_file() {
        //auto                     start = std::chrono::high_resolution_clock::now();
        mqi::vec3<ijk_t>         dim;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        printf("%d\n", this->num_spots);
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim = this->world->children[c_ind]->geo->get_nxyz();
                mqi::io::save_to_npz<R>(this->world->children[c_ind]->scorers[s_ind],
                                        this->particles_per_history,
                                        this->output_path,
                                        filename,
                                        dim,
                                        this->num_spots);
            }
        }
        //auto                                      stop = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> duration = stop - start;
        //printf("Reshape and save to npz done %f ms\n", duration.count());
    }

    CUDA_HOST
    virtual void
    save_files() {
        ///< Get node pointer for phantom as we know it's id w.r.t world
        /// Binary output only
        uint32_t                 vol_size;
        mqi::vec3<ijk_t>         dim;
        R*                       reshaped_data;
        std::string              filename;
        std::vector<std::string> beam_names = this->tx->get_beam_names();
        std::string              beam_name  = beam_names[bnb - 1];
        for (int c_ind = 0; c_ind < this->world->n_children; c_ind++) {
            for (int s_ind = 0; s_ind < this->world->children[c_ind]->n_scorers; s_ind++) {
                filename = beam_name + "_" + std::to_string(c_ind) + "_" +
                           this->world->children[c_ind]->scorers[s_ind]->name_;
                dim      = this->world->children[c_ind]->geo->get_nxyz();
                vol_size = dim.x * dim.y * dim.z;
                mqi::io::save_to_bin<R>(this->world->children[c_ind]->scorers[s_ind],
                                        this->particles_per_history,
                                        this->output_path,
                                        filename);
            }
        }
    }
};

}   // namespace mqi

#endif
