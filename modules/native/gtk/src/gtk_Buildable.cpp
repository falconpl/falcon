/**
 *  \file gtk_Buildable.cpp
 */

#include "gtk_Buildable.hpp"


#if GTK_CHECK_VERSION( 2, 12, 0 )

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void Buildable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "set_name",               &Buildable::set_name },
    { "get_name",               &Buildable::get_name },
#if 0
    { "add_child",              &Buildable::add_child },
    { "set_buildable_property", &Buildable::set_buildable_property },
    { "construct_child",        &Buildable::construct_child },
    { "custom_tag_start",       &Buildable::custom_tag_start },
    { "custom_tag_end",         &Buildable::custom_tag_end },
    { "custom_finished",        &Buildable::custom_finished },
    { "parser_finished",        &Buildable::parser_finished },
    { "get_internal_child",     &Buildable::get_internal_child },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkBuildable
    @brief Interface for objects that can be built by GtkBuilder

    In order to allow construction from a GtkBuilder UI description, an object class
    must implement the GtkBuildable interface. The interface includes methods for
    setting names and properties of objects, parsing custom tags, constructing child
    objects.

    The GtkBuildable interface is implemented by all widgets and many of the non-widget
    objects that are provided by GTK+. The main user of this interface is GtkBuilder,
    there should be very little need for applications to call any gtk_buildable_...
    functions.
 */


/*#
    @method set_name GtkBuildable
    @brief Sets the name of the buildable object.
    @param name name to set
 */
FALCON_FUNC Buildable::set_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_buildable_set_name( (GtkBuildable*)_obj, nm );
}


/*#
    @method get_name GtkBuildable
    @brief Gets the name of the buildable object.
    @return the name set with gtk_buildable_set_name()

    GtkBuilder sets the name based on the the GtkBuilder UI definition used to construct
    the buildable.
 */
FALCON_FUNC Buildable::get_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_buildable_get_name( (GtkBuildable*)_obj );
    if ( nm )
        vm->retval( new String( nm ) );
    else
        vm->retnil();
}


#if 0
FALCON_FUNC Buildable::add_child( VMARG );

FALCON_FUNC Buildable::set_buildable_property( VMARG );

FALCON_FUNC Buildable::construct_child( VMARG );

FALCON_FUNC Buildable::custom_tag_start( VMARG );

FALCON_FUNC Buildable::custom_tag_end( VMARG );

FALCON_FUNC Buildable::custom_finished( VMARG );

FALCON_FUNC Buildable::parser_finished( VMARG );

FALCON_FUNC Buildable::get_internal_child( VMARG );
#endif

} // Gtk
} // Falcon

#endif // GTK_CHECK_VERSION( 2, 12, 0 )
