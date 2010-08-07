/**
 *  \file gtk_Object.cpp
 */

#include "gtk_Object.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Object::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Object = mod->addClass( "GtkObject", &Gtk::abstract_init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_Object->getClassDef()->addInheritance( in );

    c_Object->getClassDef()->factory( &Object::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_destroy",     &Object::signal_destroy },
    { "destroy",            &Object::destroy },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Object, meth->name, meth->cb );

}


Object::Object( const Falcon::CoreClass* gen, const GtkObject* object )
    :
    Gtk::CoreGObject( gen, (GObject*) object )
{}


Falcon::CoreObject* Object::factory( const Falcon::CoreClass* gen, void* object, bool )
{
    return new Object( gen, (GtkObject*) object );
}


/*#
    @class GtkObject
    @brief The base class of the GTK+ type hierarchy

    GtkObject is the base class for all widgets, and for a few non-widget objects
    such as GtkAdjustment. GtkObject predates GObject; non-widgets that derive from
    GtkObject rather than GObject do so for backward compatibility reasons.
 */

/*#
    @method signal_destroy GtkObject
    @brief Connect a VMSlot to the object destroy signal and return it

    Signals that all holders of a reference to the GtkObject should release the
    reference that they hold. May result in finalization of the object if all
    references are released.
 */
FALCON_FUNC Object::signal_destroy( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "destroy", (void*) &Gtk::Object::on_destroy, vm );
}


void Object::on_destroy( GObject* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "destroy", "on_destroy", (VMachine*)_vm );
}


/*#
    @method destroy GtkObject
    @brief Emits the "destroy" signal.

    Important note: Falcon will internally manage destruction of objects.
    Use at your own risks.

    Emits the "destroy" signal notifying all reference holders that they should
    release the GtkObject.

    The memory for the object itself won't be deleted until its reference count
    actually drops to 0; gtk_object_destroy() merely asks reference holders to
    release their references, it does not free the object.

 */
FALCON_FUNC Object::destroy( VMARG )
{
    NO_ARGS
    gtk_object_destroy( GET_GTKOBJECT( vm->self() ) );
}


} // Gtk
} // Falcon
