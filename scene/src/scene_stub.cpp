// Compiled only when libsdformat is not available.
// Keeps the scene library target non-empty; actual APIs raise at runtime.
#ifndef DYPHUR_SCENE_ENABLED

#include <scene/sdf_loader.hpp>
#include <stdexcept>

namespace dyphur {

SceneDesc load_sdf(const std::string&, const SdfLoadParams&)
{
    throw std::runtime_error(
        "dyphur::load_sdf: scene module was built without libsdformat. "
        "Install the 'sdformat' vcpkg package and reconfigure.");
}

} // namespace dyphur

#endif // DYPHUR_SCENE_ENABLED
