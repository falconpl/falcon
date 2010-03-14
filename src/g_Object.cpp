/**
 *  \file g_Object.cpp
 */

#include "g_Object.hpp"

#include <glib-object.h>


namespace Falcon {
namespace Glib {

/**
 *  \brief module init
 */
void Object::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Object = mod->addClass( "GObject", &Gtk::abstract_init );

    Gtk::MethodTab methods[] =
    {
    { "signal_notify",      &Object::signal_notify },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Object, meth->name, meth->cb );
}

/*#
    @class gtk.GObject
    @brief The base object type

    GObject is the fundamental type providing the common attributes and methods
    for all object types in GTK+, Pango and other libraries based on GObject.
    The GObject class provides methods for object construction and destruction,
    property access methods, and signal support.
 */


/*#
    @method signal_notify gtk.GObject
    @brief Connect a VMSlot to the object notify signal and return it

    The notify signal is emitted on an object when one of its properties has been
    changed. Note that getting this signal doesn't guarantee that the value of the
    property has actually changed, it may also be emitted when the setter for the
    property is called to reinstate the previous value.
 */
FALCON_FUNC Object::signal_notify( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "notify", (void*) &Object::on_notify, vm );
}


void Object::on_notify( GObject* obj, GParamSpec* pspec, gpointer _vm )
{
    Gtk::internal_trigger_slot( obj, "notify", "on_notify", (VMachine*)_vm );
}


} // Glib
} // Falcon
