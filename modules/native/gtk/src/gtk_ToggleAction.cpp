/**
 *  \file gtk_ToggleAction.cpp
 */

#include "gtk_ToggleAction.hpp"

#include <gtk/gtk.h>

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToggleAction::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToggleAction = mod->addClass( "GtkToggleAction", &ToggleAction::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkAction" ) );
    c_ToggleAction->getClassDef()->addInheritance( in );

    c_ToggleAction->getClassDef()->factory( &ToggleAction::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",     &ToggleAction::signal_toggled },
    { "toggled",            &ToggleAction::toggled },
    { "set_active",         &ToggleAction::set_active },
    { "get_active",         &ToggleAction::get_active },
    { "set_draw_as_radio",  &ToggleAction::set_draw_as_radio },
    { "get_draw_as_radio",  &ToggleAction::get_draw_as_radio },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToggleAction, meth->name, meth->cb );
}


ToggleAction::ToggleAction( const Falcon::CoreClass* gen, const GtkToggleAction* action )
    :
    Gtk::CoreGObject( gen, (GObject*) action )
{}


Falcon::CoreObject* ToggleAction::factory( const Falcon::CoreClass* gen, void* action, bool )
{
    return new ToggleAction( gen, (GtkToggleAction*) action );
}


/*#
    @class GtkToggleAction
    @brief An action which can be toggled between two states
    @optparam name A unique name for the action
    @optparam label The label displayed in menu items and on buttons
    @optparam tooltip A tooltip for the action
    @optparam stock_id The stock icon to display in widgets representing the action

    A GtkToggleAction corresponds roughly to a GtkCheckMenuItem. It has an "active"
    state specifying whether the action has been checked or not.

    To add the action to a GtkActionGroup and set the accelerator for the action,
    call gtk_action_group_add_action_with_accel().
 */
FALCON_FUNC ToggleAction::init( VMARG )
{
    MYSELF;

    if ( self->getObject() )
        return;

    Gtk::ArgCheck4 args( vm, "S[,S,S,S]" );

    const gchar* name = args.getCString( 0 );
    const gchar* label = args.getCString( 1, false );
    const gchar* tooltip = args.getCString( 2, false );
    const gchar* stock = args.getCString( 3, false );

    GtkToggleAction* act = gtk_toggle_action_new( name, label, tooltip, stock );
    self->setObject( (GObject*) act );
}


/*#
    @method signal_toggled GtkToggleAction
    @brief Connect a VMSlot to the action toggled signal and return it
 */
FALCON_FUNC ToggleAction::signal_toggled( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "toggled", (void*) &ToggleAction::on_toggled, vm );
}


void ToggleAction::on_toggled( GtkToggleAction* act, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) act, "toggled", "on_toggled", (VMachine*)_vm );
}


/*#
    @method toggled GtkToggleAction
    @brief Emits the "toggled" signal on the toggle action.
 */
FALCON_FUNC ToggleAction::toggled( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_action_toggled( (GtkToggleAction*)_obj );
}


/*#
    @method set_active GtkToggleAction
    @brief Sets the checked state on the toggle action.
    @param is_active (boolean) whether the action should be checked or not
 */
FALCON_FUNC ToggleAction::set_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_action_set_active( (GtkToggleAction*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_active GtkToggleAction
    @brief Returns the checked state of the toggle action.
    @return the checked state of the toggle action
 */
FALCON_FUNC ToggleAction::get_active( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_action_get_active( (GtkToggleAction*)_obj ) );
}


/*#
    @method set_draw_as_radio GtkToggleAction
    @brief Sets whether the action should have proxies like a radio action.
    @param draw_as_radio (boolean) whether the action should have proxies like a radio action
 */
FALCON_FUNC ToggleAction::set_draw_as_radio( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_action_set_draw_as_radio( (GtkToggleAction*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_draw_as_radio GtkToggleAction
    @brief Returns whether the action should have proxies like a radio action.
    @return whether the action should have proxies like a radio action.
 */
FALCON_FUNC ToggleAction::get_draw_as_radio( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_action_get_draw_as_radio( (GtkToggleAction*)_obj ) );
}


} // Gtk
} // Falcon
