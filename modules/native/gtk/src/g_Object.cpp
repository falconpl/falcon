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
    { "set_property",       &Object::set_property },
    { "get_property",       &Object::get_property },
    { "notify",             &Object::notify },
    { "freeze_notify",      &Object::freeze_notify },
    { "thaw_notify",        &Object::thaw_notify },
    { "ref",                &Object::ref },
    { "unref",              &Object::unref },
    { "ref_sink",           &Object::ref_sink },
#if 0 // todo?
    { "connect",            &Object::connect },
    { "disconnect",         &Object::disconnect },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Object, meth->name, meth->cb );
}


Object::Object( const Falcon::CoreClass* gen, const GObject* obj )
    :
    Gtk::CoreGObject( gen, obj )
{}


Falcon::CoreObject* Object::factory( const Falcon::CoreClass* gen, void* obj, bool )
{
    return new Object( gen, (GObject*) obj );
}

/*#
   @beginmodule gtk
*/

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
    @brief The notify signal is emitted on an object when one of its properties has been changed.

    Note that getting this signal doesn't guarantee that the value of the
    property has actually changed, it may also be emitted when the setter for the
    property is called to reinstate the previous value.
 */
FALCON_FUNC Object::signal_notify( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "notify", (void*) &Object::on_notify, vm );
}


void Object::on_notify( GObject* obj, GParamSpec* pspec, gpointer _vm )
{
    CoreGObject::trigger_slot( obj, "notify", "on_notify", (VMachine*)_vm );
}


/*#
    @method set_property GObject
    @brief Sets a property on an object.
    @param property_name name of the property to set
    @param value value for the property

    @code
    w = GtkWindow()
    w.set_property( "title", "Falcon" )
    @endcode
 */
FALCON_FUNC Object::set_property( VMARG )
{
    Item* i_name = vm->param( 0 );
    Item* i_val = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString()
        || !i_val )
        throw_inv_params( "S,X" );
#endif
    AutoCString name( i_name->asString() );
    GObject* gobj = GET_OBJECT( vm->self() );

    switch ( i_val->type() )
    {
    case FLC_ITEM_NIL:
        g_object_set( gobj, name.c_str(), NULL, NULL );
        break;
    case FLC_ITEM_INT:
        g_object_set( gobj, name.c_str(), i_val->asInteger(), NULL );
        break;
    case FLC_ITEM_BOOL:
        g_object_set( gobj, name.c_str(), i_val->asBoolean(), NULL );
        break;
    case FLC_ITEM_NUM:
        g_object_set( gobj, name.c_str(), i_val->asNumeric(), NULL );
        break;
    case FLC_ITEM_STRING:
    {
        AutoCString val( i_val->asString() );
        g_object_set( gobj, name.c_str(), val.c_str(), NULL );
        break;
    }
    case FLC_ITEM_OBJECT:
    {
        VoidObject* vobj = dynamic_cast<Gtk::VoidObject*>( i_val->asObjectSafe() );
        if ( vobj != 0 )
            g_object_set( gobj, name.c_str(), vobj->getObject(), NULL );
        else
            throw_inv_params( "S,GTK" );
        break;
    }
    default:
        throw_inv_params( "not implemented" );
    }
}


/*#
    @method get_property GObject
    @brief Gets a property of an object.
    @param property_name (string)
    @return the property value
 */
FALCON_FUNC Object::get_property( VMARG )
{
    Item* i_name = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_name || !i_name->isString() )
        throw_inv_params( "S" );
#endif
    GObject* gobj = GET_OBJECT( vm->self() );
    AutoCString name( i_name->asString() );

    GParamSpec* spec = g_object_class_find_property(
        G_OBJECT_GET_CLASS( gobj ), name.c_str() );

    if ( !spec )
        throw_gtk_error( e_inv_property, FAL_STR( gtk_e_inv_property_ ) );

    switch ( spec->value_type )
    {
    case G_TYPE_NONE:
        vm->retnil();
        return;
    case G_TYPE_BOOLEAN:
    {
        gboolean b;
        g_object_get( gobj, name.c_str(), &b, NULL );
        vm->retval( (bool) b );
        return;
    }
    case G_TYPE_INT:
    case G_TYPE_UINT:
    case G_TYPE_LONG:
    case G_TYPE_ULONG:
    case G_TYPE_INT64:
    case G_TYPE_UINT64:
    {
        Falcon::int64 i = 0;
        g_object_get( gobj, name.c_str(), &i, NULL );
        vm->retval( i );
        return;
    }
    case G_TYPE_FLOAT:
    case G_TYPE_DOUBLE:
    {
        double d = 0;
        g_object_get( gobj, name.c_str(), &d, NULL );
        vm->retval( d );
        return;
    }
    case G_TYPE_STRING:
    {
        gchar* txt = NULL;
        g_object_get( gobj, name.c_str(), &txt, NULL );
        if ( txt )
        {
            vm->retval( UTF8String( txt ) );
            g_free( txt );
        }
        else
            vm->retval( UTF8String( "" ) );
        return;
    }
    default:
    {
        switch ( G_TYPE_FUNDAMENTAL( spec->value_type ) )
        {
        case G_TYPE_ENUM:
        {
            Falcon::int64 val = 0;
            g_object_get( gobj, name.c_str(), &val, NULL );
            vm->retval( val );
            return;
        }
        case G_TYPE_OBJECT:
        {
            void* o;
            g_object_get( gobj, name.c_str(), &o, NULL );
            Gtk::CoreGObject::retval( vm, spec->value_type, o );
            return;
        }
        case G_TYPE_BOXED:
        {
            void* o;
            g_object_get( gobj, name.c_str(), &o, NULL );
            Gtk::VoidObject::retval( vm, spec->value_type, o );
            return;
        }
        default:
            g_print( "type name %s\n"
                     "GType %ld\n"
                     "Fundamental %s\n",
                g_type_name( G_PARAM_SPEC_VALUE_TYPE( spec ) ),
                spec->value_type,
                g_type_name( G_TYPE_FUNDAMENTAL( spec->value_type ) ) );

            throw_inv_params( "not yet implemented..." );
        }
    }
    }
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
    if ( !i_nam || !i_nam->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString name( i_nam->asString() );
    g_object_notify( GET_OBJECT( vm->self() ), name.c_str() );
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
    NO_ARGS
    g_object_freeze_notify( GET_OBJECT( vm->self() ) );
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
    NO_ARGS
    g_object_thaw_notify( GET_OBJECT( vm->self() ) );
}


/*#
    @method ref GObject
    @brief Increases the reference count of object.
    @note Don't use this...
 */
FALCON_FUNC Object::ref( VMARG )
{
    NO_ARGS
    g_object_ref( GET_OBJECT( vm->self() ) );
}


/*#
    @method unref GObject
    @brief Decreases the reference count of object.
    @note Don't use this...

    When its reference count drops to 0, the object is finalized (i.e. its memory is freed).
 */
FALCON_FUNC Object::unref( VMARG )
{
    NO_ARGS
    g_object_unref( GET_OBJECT( vm->self() ) );
}


/*#
    @method ref_sink GObject
    @brief Increase the reference count of object, and possibly remove the floating reference, if object has a floating reference.
    @note Don't use this...

    In other words, if the object is floating, then this call "assumes ownership"
    of the floating reference, converting it to a normal reference by clearing the
    floating flag while leaving the reference count unchanged. If the object is not
    floating, then this call adds a new normal reference increasing the reference count by one.
 */
FALCON_FUNC Object::ref_sink( VMARG )
{
    NO_ARGS
    g_object_ref_sink( GET_OBJECT( vm->self() ) );
}


} // Glib
} // Falcon
