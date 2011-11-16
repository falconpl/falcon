/**
 *  \file gtk_AccelGroup.cpp
 */

#include "gtk_AccelGroup.hpp"


/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void AccelGroup::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_AccelGroup = mod->addClass( "GtkAccelGroup", &AccelGroup::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GObject" ) );
    c_AccelGroup->getClassDef()->addInheritance( in );

    c_AccelGroup->setWKS( true );
    c_AccelGroup->getClassDef()->factory( &AccelGroup::factory );

    Gtk::MethodTab methods[] =
    {
    { "connect_group",      AccelGroup::connect_group },
#if 0
    { "connect_by_path",    AccelGroup::connect_by_path },
    { "disconnect",         AccelGroup::disconnect },
    { "disconnect_key",     AccelGroup::disconnect_key },
    { "query",              AccelGroup::query },
    { "activate",           AccelGroup::activate },
    { "lock",               AccelGroup::lock },
    { "unlock",             AccelGroup::unlock },
    { "get_is_locked",      AccelGroup::get_is_locked },
    { "from_accel_closure", AccelGroup::from_accel_closure },
    { "get_modifier_mask",  AccelGroup::get_modifier_mask },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_AccelGroup, meth->name, meth->cb );
}


AccelGroup::AccelGroup( const Falcon::CoreClass* gen, const GtkAccelGroup* acc )
    :
    Gtk::CoreGObject( gen, (GObject*) acc )
{}


Falcon::CoreObject* AccelGroup::factory( const Falcon::CoreClass* gen, void* acc, bool )
{
    return new AccelGroup( gen, (GtkAccelGroup*) acc );
}


/*#
    @class GtkAccelGroup
    @brief Groups of global keyboard accelerators for an entire GtkWindow

    A GtkAccelGroup represents a group of keyboard accelerators, typically
    attached to a toplevel GtkWindow (with gtk_window_add_accel_group()).
    Usually you won't need to create a GtkAccelGroup directly; instead, when
    using GtkItemFactory, GTK+ automatically sets up the accelerators for your
    menus in the item factory's GtkAccelGroup.

    Note that accelerators are different from mnemonics. Accelerators are shortcuts
    for activating a menu item; they appear alongside the menu item they're a
    shortcut for. For example "Ctrl+Q" might appear alongside the "Quit" menu
    item. Mnemonics are shortcuts for GUI elements such as text entries or
    buttons; they appear as underlined characters. See gtk_label_new_with_mnemonic().
    Menu items can have both accelerators and mnemonics, of course.
 */
FALCON_FUNC AccelGroup::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;
    self->setObject( (GObject*) gtk_accel_group_new() );
}


gboolean AccelGroup::activate_cb( GtkAccelGroup* acc, GObject* obj,
                                  guint keyval, GdkModifierType modifier,
                                  gpointer lock )
{
    VMachine* vm = VMachine::getCurrent();
    Item cb = ((GarbageLock*)lock)->item();

    vm->pushParam( 0 ); // push a dummy 0 until we can get the right object
    vm->pushParam( (int64) keyval );
    vm->pushParam( (int64) modifier );
    vm->callItem( cb, 3 );
    Item it = vm->regA();

    if ( !it.isNil() && it.isBoolean() )
    {
        return it.asBoolean() ? TRUE : FALSE;
    }
    else
    {
        printf(
            "[GtkAccelGroup::activate_cb] invalid callback (expected boolean)\n" );
        return FALSE; // unhandled event
    }
}


/*#
    @method connect_group
    @brief Installs an accelerator in this group.
    @param accel_key key of the accelerator
    @param accel_mods modifier combination of the accelerator
    @param accel_flags a flag mask to configure this accelerator
    @param callback a function or method to be executed upon accelerator activation

    The connect_group() method installs an accelerator in the accelerator group.
    When the accelerator group is being activated, the function (or method)
    specified by callback will be invoked if the accelerator key and modifier key
    match those specified by accel_key and accel_mods.

    The value of modifier is a combination of the GDK Modifier Constants.
    accel_flags is a combination of gtk.ACCEL_VISIBLE and gtk.ACCEL_LOCKED.

    The callback function is defined as:

    function callback(acceleratable, keyval, modifier)

    where acceleratable is the object that
    the accel_group is attached to (e.g. a gtk.Window), keyval is the accelerator
    key and modifier is the key modifier. callback returns True if the accelerator
    was handled by callback.
 */
FALCON_FUNC AccelGroup::connect_group( VMARG )
{
    Item* i_key = vm->param( 0 );
    Item* i_mods = vm->param( 1 );
    Item* i_flags = vm->param( 2 );
    Item* i_cb = vm->param( 3 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_key || !i_key->isString()
        || !i_mods || !i_mods->isInteger()
        || !i_flags || !i_flags->isInteger()
        || !i_cb || !( i_cb->isCallable() || i_cb->isComposed() ) )
        throw_inv_params( "S,GdkModifierType,GtkAccelFlags,C" );
#endif
    MYSELF;
    GET_OBJ( self );
    String* chr = i_key->asString();
    guint keyval = chr->length() ? chr->getCharAt( 0 ) : 0;

    GarbageLock* lock = CoreGObject::lockItem( _obj, *i_cb );
    GClosure* cl = g_cclosure_new( G_CALLBACK( &AccelGroup::activate_cb ),
                                   (gpointer) lock,
                                   NULL );
    g_object_watch_closure( _obj, cl );

    gtk_accel_group_connect( GET_ACCELGROUP( vm->self() ),
                             keyval,
                             (GdkModifierType) i_mods->asInteger(),
                             (GtkAccelFlags) i_flags->asInteger(),
                             cl );
}


} // Gtk
} // Falcon

// vi: set ai et sw=4 ts=4 sts=4:
// kate: replace-tabs on; shift-width 4;
