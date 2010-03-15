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
    { "set",                &Object::set },
    { "notify",             &Object::notify },
    { "freeze_notify",      &Object::freeze_notify },
    { "thaw_notify",        &Object::thaw_notify },
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


/*#
    @method set gtk.GObject
    @brief Sets a property on an object.
    @param property_name name of the property to set
    @param value value for the property (nil, integer, boolean, numeric, or string)
 */
FALCON_FUNC Object::set( VMARG )
{
    Item* i_nam = vm->param( 0 );
    Item* i_val = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nam || i_nam->isNil() || !i_nam->isString()
        || !i_val )
        throw_inv_params( "S,?" );
#endif
    AutoCString name( i_nam->asString() );
    MYSELF;
    GET_OBJ( self );

    if ( i_val->isNil() )
        g_object_set( _obj, name.c_str(), NULL, NULL );
    else
    if ( i_val->isInteger() )
        g_object_set( _obj, name.c_str(), i_val->asInteger(), NULL );
    else
    if ( i_val->isBoolean() )
        g_object_set( _obj, name.c_str(), i_val->asBoolean(), NULL );
    else
    if ( i_val->isNumeric() )
        g_object_set( _obj, name.c_str(), i_val->asNumeric(), NULL );
    else
    if ( i_val->isString() )
    {
        AutoCString val( i_val->asString() );
        g_object_set( _obj, name.c_str(), val.c_str(), NULL );
    }
#if 0
    else
    if ( i_val->isObject() )
    {
        void* obj = i_val->asObject()->getUserData();
#ifndef NO_PARAMETER_CHECK
        if ( !obj )
            throw_inv_params( "No object.." );
#endif
        g_object_set( _obj, name.c_str(), obj, NULL );
    }
#endif
    else
        throw_inv_params( "S,?" );
}


/*#
    @method notify gtk.GObject
    @brief Emits a "notify" signal for the property property_name on object.
    @param property_name (string)
 */
FALCON_FUNC Object::notify( VMARG )
{
    Item* i_nam = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nam || i_nam->isNil() || !i_nam->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_nam->asString() );
    MYSELF;
    GET_OBJ( self );
    g_object_notify( _obj, name.c_str() );
}


/*#
    @method freeze_notify gtk.GObject
    @brief Increases the freeze count on object.

    If the freeze count is non-zero, the emission of "notify" signals on object
    is stopped. The signals are queued until the freeze count is decreased to zero.

    This is necessary for accessors that modify multiple properties to prevent
    premature notification while the object is still being modified.
 */
FALCON_FUNC Object::freeze_notify( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    g_object_freeze_notify( _obj );
}


/*#
    @method thaw_notify gtk.GObject
    @brief Reverts the effect of a previous call to freeze_notify().

    The freeze count is decreased on object and when it reaches zero, all queued
    "notify" signals are emitted.

    It is an error to call this function when the freeze count is zero.
 */
FALCON_FUNC Object::thaw_notify( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    g_object_thaw_notify( _obj );
}


} // Glib
} // Falcon
