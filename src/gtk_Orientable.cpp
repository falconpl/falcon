/**
 *  \file gtk_Orientable.cpp
 */

#include "gtk_Orientable.hpp"


#if GTK_MINOR_VERSION >= 16

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void Orientable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "get_orientation",    &Orientable::get_orientation },
    { "set_orientation",    &Orientable::set_orientation },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkOrientable
    @brief An interface for flippable widgets

    The GtkOrientable interface is implemented by all widgets that can be oriented
    horizontally or vertically.

    Historically, such widgets have been realized as subclasses of a common base
    class (e.g GtkBox/GtkHBox/GtkVBox and GtkScale/GtkHScale/GtkVScale).
    GtkOrientable is more flexible in that it allows the orientation to be changed
    at runtime, allowing the widgets to 'flip'.
 */


/*#
    @method get_orientation GtkOrientable
    @brief Retrieves the orientation of the orientable.
    @return the orientation of the orientable.
 */
FALCON_FUNC Orientable::get_orientation( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_orientable_get_orientation( (GtkOrientable*)_obj ) );
}


/*#
    @method set_orientation GtkOrientable
    @brief Sets the orientation of the orientable.
    @param orientation the orientable's new orientation.
 */
FALCON_FUNC Orientable::set_orientation( VMARG )
{
    Item* i_orient = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_orient || i_orient->isNil() || !i_orient->isInteger() )
        throw_inv_params( "GtkOrientation" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_orientable_set_orientation(
            (GtkOrientable*)_obj, (GtkOrientation) i_orient->asInteger() );
}


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 16
