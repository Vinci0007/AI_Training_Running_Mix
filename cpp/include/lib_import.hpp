// base on system platform , import libs
// author: tourist
// date: 2021.06.01

// windows platform import libs

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <string>
#include <vector>
#include <filesystem>

std::string path = "..\\..\\..\\3rd_party";
std::vector<std::string> findDLLsInDirectory(const std::string& directory) {
    std::vector<std::string> dlls;
    try {
        std::filesystem::path dirPath(directory);
        if (std::filesystem::exists(dirPath) && std::filesystem::is_directory(dirPath)) {
            std::filesystem::recursive_directory_iterator it(dirPath), end;
            for (; it != end; ++it) {
                if (it->path().extension() == ".dll") {
                    dlls.push_back(it->path().string());
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        // Handle any filesystem errors
        OutputDebugStringA(e.what());
    }
    return dlls;
}

// #define LIB_IMPORT(lib_name) HMODULE lib_handle = LoadLibraryA(#lib_name ".dll")

#define LIB_IMPORT(lib_name) \
    std::vector<std::string> dlls = findDLLsInDirectory(path); \
    HMODULE lib_handle = NULL; \
    for (const auto& dllPath : dlls) { \
        if (std::string(dllPath).find(#lib_name ".dll") != std::string::npos) { \
            lib_handle = LoadLibraryExA(dllPath.c_str(), NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS); \
            if (lib_handle != NULL) { \
                break; \
            } \
        } \
    } \
    if (lib_handle == NULL) { \
        // 处理加载失败的情况 \
        DWORD error = GetLastError(); \
        // 输出错误信息 \
    }
#define LIB_FUNC(func_name) GetProcAddress(lib_handle, #func_name)
#define LIB_RELEASE() FreeLibrary(lib_handle)
#endif  // _WIN32 || _WIN64

// linux platform import libs
#if defined(__linux__)
#include <dlfcn.h>

std::string path = "../../../3rd_party";  // 使用相对路径

#define LIB_IMPORT(lib_name) \
    std::vector<std::string> soFiles = findLibsInDirectory(path, ".so"); \
    void* lib_handle = NULL; \
    for (const auto& soPath : soFiles) { \
        if (std::string(soPath).find(#lib_name ".so") != std::string::npos) { \
            lib_handle = dlopen(soPath.c_str(), RTLD_LAZY); \
            if (lib_handle != NULL) { \
                break; \
            } \
        } \
    } \
    if (lib_handle == NULL) { \
        // 处理加载失败的情况 \
        const char* error = dlerror(); \
        // 输出错误信息 \
    }

#define LIB_FUNC(func_name) dlsym(lib_handle, #func_name)
#define LIB_RELEASE() dlclose(lib_handle)
#endif  // __linux__

// macos platform import libs
#if defined(__APPLE__)
#include <dlfcn.h>

std::string path = "../../../3rd_party";  // 使用相对路径

#define LIB_IMPORT(lib_name) \
    std::vector<std::string> dylibFiles = findLibsInDirectory(path, ".dylib"); \
    void* lib_handle = NULL; \
    for (const auto& dylibPath : dylibFiles) { \
        if (std::string(dylibPath).find(#lib_name ".dylib") != std::string::npos) { \
            lib_handle = dlopen(dylibPath.c_str(), RTLD_LAZY); \
            if (lib_handle != NULL) { \
                break; \
            } \
        } \
    } \
    if (lib_handle == NULL) { \
        // 处理加载失败的情况 \
        const char* error = dlerror(); \
        // 输出错误信息 \
    }

#define LIB_FUNC(func_name) dlsym(lib_handle, #func_name)
#define LIB_RELEASE() dlclose(lib_handle)
#endif  // __APPLE__

// common platform import libs, no need to recognizer platform

// #define LIB_IMPORT(lib_name) void* lib_handle = dlopen(#lib_name ".so", RTLD_LAZY)
// #define LIB_FUNC(func_name) dlsym(lib_handle, #func_name)
// #define LIB_RELEASE() dlclose(lib_handle)



