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

    c_Object->setWKS( true );
    c_Object->getClassDef()->factory( &Object::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_notify",      &Object::signal_notify },
    { "set",                &Object::set },
    { "get",                &Object::get },
    { "notify",             &Object::notify },
    { "freeze_notify",      &Object::freeze_notify },
    { "thaw_notify",        &Object::thaw_notify },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Object, meth->name, meth->cb );
}


Object::Object( const Falcon::CoreClass* gen, const GObject* obj )
    :
    Gtk::CoreGObject( gen )
{
    if ( obj )
        setUserData( new Gtk::GData( (GObject*) obj ) );
}


Falcon::CoreObject* Object::factory( const Falcon::CoreClass* gen, void* obj, bool )
{
    return new Object( gen, (GObject*) obj );
}


/*#
    @class GObject
    @brief The base object type

    GObject is the fundamental type providing the common attributes and methods
    for all object types in GTK+, Pango and other libraries based on GObject.
    The GObject class provides methods for object construction and destruction,
    property access methods, and signal support.
 */


/*#
    @method signal_notify GObject
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
    @method set GObject
    @brief Sets a property on an object.
    @param property_name name of the property to set
    @param value value for the property (nil, integer, boolean, numeric, or string)

    @code
    w = GtkWindow()
    w.set( "title", "Falcon" )
    @endcode
 */
FALCON_FUNC Object::set( VMARG )
{
    Item* i_nam = vm->param( 0 );
    Item* i_val = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nam || i_nam->isNil() || !i_nam->isString()
        || !i_val )
        throw_inv_params( "S,X" );
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
    else
    if ( i_val->isObject() )
    {
#ifndef NO_PARAMETER_CHECK
        if ( !IS_DERIVED( i_val, GObject ) )
            throw_inv_params( "GObject" );
#endif
        GObject* obj = (GObject*)((Gtk::GData*)i_val->asObject()->getUserData())->obj();
        g_object_set( _obj, name.c_str(), obj, NULL );
    }
    else
        throw_inv_params( "S,X" );
}


/*#
    @method get GObject
    @brief Gets a property of an object.
    @param property_name (string)
    @return the property value
 */
FALCON_FUNC Object::get( VMARG )
{
    Item* i_nam = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_nam || !i_nam->isString() )
        throw_inv_params( "S" );
#endif
    MYSELF;
    GET_OBJ( self );
    AutoCString nam( i_nam->asString() );

    GParamSpec* spec = g_object_class_find_property(
        G_OBJECT_GET_CLASS( _obj ), nam.c_str() );

    if ( !spec )
        throw_gtk_error( e_inv_property, FAL_STR( gtk_e_inv_property_ ) );

    if ( spec->value_type == G_TYPE_BOOLEAN )
    {
        gboolean b;
        g_object_get( _obj, nam.c_str(), &b, NULL );
        vm->retval( (bool) b );
    }
    else
    if (   spec->value_type == G_TYPE_INT
        || spec->value_type == G_TYPE_UINT
        || spec->value_type == G_TYPE_LONG
        || spec->value_type == G_TYPE_ULONG
        || spec->value_type == G_TYPE_INT64
        || spec->value_type == G_TYPE_UINT64 )
    {
        Falcon::int64 val;
        g_object_get( _obj, nam.c_str(), &val, NULL );
        vm->retval( val );
    }
    else
    if (   spec->value_type == G_TYPE_FLOAT
        || spec->value_type == G_TYPE_DOUBLE )
    {
        double d;
        g_object_get( _obj, nam.c_str(), &d, NULL );
        vm->retval( d );
    }
    else
    if ( spec->value_type == G_TYPE_STRING )
    {
        gchar* txt;
        g_object_get( _obj, nam.c_str(), &txt, NULL );
        String* s;
        if ( txt )
        {
            s = new String( txt );
            s->bufferize();
            g_free( txt );
        }
        else
            s = new String;
        vm->retval( s );
    }
    else
    if ( g_type_is_a( spec->value_type, G_TYPE_OBJECT ) )
    {
        GObject* o;
        g_object_get( _obj, nam.c_str(), &o, NULL );
        if ( o )
        {
            Item* wki = vm->findWKI( "GObject" );
            vm->retval( new Object( wki->asClass(), o ) );
            g_object_unref( o );
        }
        else
            vm->retnil();
    }
    else
        throw_inv_params( "not implemented" );
}


/*#
    @method notify GObject
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
    @method freeze_notify GObject
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
    @method thaw_notify GObject
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
