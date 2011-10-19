/**
 *  \file gtk_Accelerator.cpp
 */

#include "gtk_Accelerator.hpp"


/*#
    @beginmodule gtk
 */

namespace Falcon {
namespace Gtk {
namespace Accelerator {

/**
 *  \brief module init
 */
void modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Accelerator = mod->addClass( "GtkAccelerator" );

    Gtk::MethodTab methods[] =
    {
    { "parse",              Accelerator::parse },
    // todo...
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Accelerator, meth->name, meth->cb );
}


/*#
    @class GtkAccelerator
    @brief Pseudo class containing static methods concerning accelerators.
 */

/*#
    @method parse GtkAccelerator
    @brief Parses a string representing an accelerator.
    @return An array with [ key, mods ]

    The format looks like "<Control>a" or "<Shift><Alt>F1" or "<Release>z" (the
    last one is for key release). The parser is fairly liberal and allows lower
    or upper case, and also abbreviations such as "<Ctl>" and "<Ctrl>". Key names
    are parsed using gdk_keyval_from_name(). For character keys the name is not
    the symbol, but the lowercase name, e.g. one would use "<Ctrl>minus" instead
    of "<Ctrl>-".
 */
FALCON_FUNC parse( VMARG )
{
    Item* i_accel = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_accel || !i_accel->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString accel( i_accel->asString() );
    guint key;
    GdkModifierType mods;
    gtk_accelerator_parse( accel.c_str(), &key, &mods );
    if ( key == 0 && mods == 0 )
        throw_inv_params( accel.c_str() ); // todo: better description?
    CoreArray* arr = new CoreArray( 2 );
    arr->append( (int64) key );
    arr->append( (int64) mods );
    vm->retval( arr );
}


} // Accelerator
} // Gtk
} // Falcon

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
