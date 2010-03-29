/**
 *  \file gtk_RadioAction.cpp
 */

#include "gtk_RadioAction.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void RadioAction::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_RadioAction = mod->addClass( "GtkRadioAction", &RadioAction::init );

    c_RadioAction->setWKS( true );
    c_RadioAction->getClassDef()->factory( &RadioAction::factory );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToggleAction" ) );
    c_RadioAction->getClassDef()->addInheritance( in );

    Gtk::MethodTab methods[] =
    {
    { "signal_changed",         &RadioAction::signal_changed },
    //{ "get_group"
    //{ "set_group"
    { "get_current_value",      &RadioAction::get_current_value },
    { "set_current_value",      &RadioAction::set_current_value },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_RadioAction, meth->name, meth->cb );
}


RadioAction::RadioAction( const Falcon::CoreClass* gen, const GtkRadioAction* act )
    :
    Gtk::CoreGObject( gen )
{
    if ( act )
        setUserData( new GData( Gtk::internal_add_slot( (GObject*) act ) ) );
}


Falcon::CoreObject* RadioAction::factory( const Falcon::CoreClass* gen, void* act, bool )
{
    return new RadioAction( gen, (GtkRadioAction*) act );
}


/*#
    @class GtkRadioAction
    @brief An action of which only one in a group can be active
    @param name A unique name for the action
    @optparam label The label displayed in menu items and on buttons
    @optparam tooltip A tooltip for the action
    @optparam stock_id The stock icon to display in widgets representing the action
    @param value (integer) The value which get_current_value() should return if this action is selected.

    A GtkRadioAction is similar to GtkRadioMenuItem. A number of radio actions can
    be linked together so that only one may be active at any one time.

    To add the action to a GtkActionGroup and set the accelerator for the action,
    call gtk_action_group_add_action_with_accel().
 */
FALCON_FUNC RadioAction::init( VMARG )
{
    MYSELF;

    Gtk::ArgCheck4 args( vm, "S[,S,S,S]" );

    const gchar* name = args.getCString( 0 );
    const gchar* label = args.getCString( 1, false );
    const gchar* tooltip = args.getCString( 2, false );
    const gchar* stock = args.getCString( 3, false );
    gint value = args.getInteger( 4 );

    GtkRadioAction* act = gtk_radio_action_new( name, label, tooltip, stock, value );
    Gtk::internal_add_slot( (GObject*) act );
    self->setUserData( new GData( (GObject*) act ) );
}


/*#
    @method signal_changed GtkRadioAction
    @brief Connect a VMSlot to the action changed signal and return it

    The changed signal is emitted on every member of a radio group when the active
    member is changed. The signal gets emitted after the activate signals for the
    previous and current active members.

    The callback function gets a GtkRadioAction as argument, that is the member
    of actions group which has just been activated.
 */
FALCON_FUNC RadioAction::signal_changed( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    Gtk::internal_get_slot( "changed", (void*) &RadioAction::on_changed, vm );
}


void RadioAction::on_changed( GtkRadioAction* obj, GtkRadioAction* current, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "changed", false );

    if ( !cs || cs->empty() )
        return;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;
    Item* wki = vm->findWKI( "GtkRadioAction" );

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_changed", it ) )
            {
                printf(
                "[GtkRadioAction::on_changed] invalid callback (expected callable)\n" );
                return;
            }
        }
        vm->pushParam( new RadioAction( wki->asClass(), current ) );
        vm->callItem( it, 1 );
    }
    while ( iter.hasCurrent() );
}


// FALCON_FUNC RadioAction::get_group( VMARG );

// FALCON_FUNC RadioAction::set_group( VMARG );


/*#
    @method get_current_value GtkRadioAction
    @brief Obtains the value property of the currently active member of the group to which action belongs.
    @return The value of the currently active group member
 */
FALCON_FUNC RadioAction::get_current_value( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_radio_action_get_current_value( (GtkRadioAction*)_obj ) );
}


/*#
    @method set_current_value GtkRadioAction
    @brief Sets the currently active group member to the member with value property current_value.
    @param current_value the new value
 */
FALCON_FUNC RadioAction::set_current_value( VMARG )
{
    Item* i_val = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_val || i_val->isNil() || !i_val->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_radio_action_set_current_value( (GtkRadioAction*)_obj, i_val->asInteger() );
}


} // Gtk
} // Falcon
