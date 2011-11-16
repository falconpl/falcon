/**
 *  \file gtk_AccelMap.cpp
 */

#include "gtk_AccelMap.hpp"


/*#
   @beginmodule gtk
 */

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void AccelMap::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_AccelMap = mod->addClass( "GtkAccelMap" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_AccelMap->getClassDef()->addInheritance( in );

    //c_AccelMap->setWKS( true );
    c_AccelMap->getClassDef()->factory( &AccelMap::factory );

    Gtk::MethodTab methods[] =
    {
    { "add_entry",          AccelMap::add_entry },

    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_AccelMap, meth->name, meth->cb );
}


AccelMap::AccelMap( const Falcon::CoreClass* gen, const GtkAccelMap* acc )
    :
    Gtk::CoreGObject( gen, (GObject*) acc )
{}


Falcon::CoreObject* AccelMap::factory( const Falcon::CoreClass* gen, void* acc, bool )
{
    return new AccelMap( gen, (GtkAccelMap*) acc );
}


/*#
    @class GtkAccelMap
    @brief Loadable keyboard accelerator specifications
 */


/*#
    @method add_entry GtkAccelMap
    @brief Registers a new accelerator with the global accelerator map.

    This function should only be called once per accel_path with the canonical
    accel_key and accel_mods for this path. To change the accelerator during
    runtime programatically, use gtk_accel_map_change_entry(). The accelerator
    path must consist of "<WINDOWTYPE>/Category1/Category2/.../Action", where
    <WINDOWTYPE> should be a unique application-specific identifier, that
    corresponds to the kind of window the accelerator is being used in, e.g.
    "Gimp-Image", "Abiword-Document" or "Gnumeric-Settings". The
    Category1/.../Action portion is most appropriately chosen by the action the
    accelerator triggers, i.e. for accelerators on menu items, choose the item's
    menu path, e.g. "File/Save As", "Image/View/Zoom" or "Edit/Select All". So
    a full valid accelerator path may look like: "<Gimp-Toolbox>/File/Dialogs/Tool Options...".
 */
FALCON_FUNC AccelMap::add_entry( VMARG )
{
    Item* i_path = vm->param( 0 );
    Item* i_key = vm->param( 1 );
    Item* i_mod = vm->param( 2 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_path || !i_path->isString()
        || !i_key || !i_key->isString()
        || !i_mod || !i_mod->isInteger() )
        throw_inv_params( "S,S,GdkModifierType" );
#endif
    AutoCString path( *i_path );
    String* chr = i_key->asString();
    guint keyval = chr->length() ? chr->getCharAt( 0 ) : 0;
    gtk_accel_map_add_entry( path.c_str(), keyval, (GdkModifierType)i_mod->asInteger() );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
