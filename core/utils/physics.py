def create_physics_material(material_prim_path: str, dynamic_friction: float = 0.5, static_friction: float = 0.5,
                            restitution: float = 0.8):
    from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
    from core.utils.usd import get_or_apply_api
    from pxr import UsdPhysics

    assert is_prim_path_valid(material_prim_path), f"Invalid prim path: {material_prim_path}"

    physics_material_api = get_or_apply_api(get_prim_at_path(material_prim_path), UsdPhysics.MaterialAPI)

    physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material_api.CreateStaticFrictionAttr(static_friction)
    physics_material_api.CreateRestitutionAttr(restitution)


def set_physics_properties(target_prim_path: str, dynamic_friction: float = 0.5, static_friction: float = 0.5,
                           restitution: float = 0.8) -> bool:
    from omni.isaac.core.utils.prims import get_prim_at_path, define_prim

    target_prim = get_prim_at_path(target_prim_path)

    assert target_prim.IsValid(), f"Invalid target_prim: {target_prim_path}"

    from core.utils.usd import get_or_apply_api, get_or_define_material
    from pxr import UsdPhysics, UsdShade

    material_prim_path = f"{target_prim_path}/physics_material"
    material_prim = define_prim(material_prim_path, "Material")

    # if the material has already been bound to the target prim, just update the physics properties
    if material_prim.HasAPI(UsdShade.MaterialBindingAPI):
        physics_material_api = get_or_apply_api(material_prim, UsdPhysics.MaterialAPI)
        physics_material_api.CreateDynamicFrictionAttr(dynamic_friction)
        physics_material_api.CreateStaticFrictionAttr(static_friction)
        physics_material_api.CreateRestitutionAttr(restitution)

        return True

    # otherwise, bind the material to the target prim
    material = get_or_define_material(material_prim_path)

    material_binding_api = get_or_apply_api(target_prim, UsdShade.MaterialBindingAPI)
    material_binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                              materialPurpose="physics")

    return True
