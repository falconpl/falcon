/**
 *  \file gtk_Activatable.cpp
 */

#include "gtk_Activatable.hpp"

#include "gtk_Action.hpp"

#include <gtk/gtk.h>

#if GTK_MINOR_VERSION >= 16

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void Activatable::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "do_set_related_action",      &Activatable::do_set_related_action },
    { "get_related_action",         &Activatable::get_related_action },
    { "get_use_action_appearance",  &Activatable::get_use_action_appearance },
    { "sync_action_properties",     &Activatable::sync_action_properties },
    { "set_related_action",         &Activatable::set_related_action },
    { "set_use_action_appearance",  &Activatable::set_use_action_appearance },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkActivatable
    @brief An interface for activatable widgets

    GtkActivatable is implemented by GtkTearoffMenuItem, GtkOptionMenu, GtkLinkButton,
    GtkRadioButton, GtkRadioToolButton, GtkToggleButton, GtkSeparatorMenuItem,
    GtkImageMenuItem, GtkCheckButton, GtkRadioMenuItem, GtkButton, GtkMenuToolButton,
    GtkMenuItem, GtkRecentChooserMenu, GtkToolItem, GtkSeparatorToolItem, GtkCheckMenuItem,
    GtkToolButton, GtkFontButton, GtkScaleButton, GtkToggleToolButton, GtkVolumeButton
    and GtkColorButton.

    Activatable widgets can be connected to a GtkAction and reflects the state of
    its action. A GtkActivatable can also provide feedback through its action, as
    they are responsible for activating their related actions.
 */


/*#
    @method do_set_related_action GtkActivatable
    @brief This is a utility function for GtkActivatable implementors.
    @param action the GtkAction to set

    When implementing GtkActivatable you must call this when handling changes of
    the "related-action", and you must also use this to break references in GObject->dispose().

    This function adds a reference to the currently set related action for you,
    it also makes sure the GtkActivatable->update() method is called when the related
    GtkAction properties change and registers to the action's proxy list.

    Note: Be careful to call this before setting the local copy of the GtkAction
    property, since this function uses gtk_activatable_get_action() to retrieve
    the previous action.

 */
FALCON_FUNC Activatable::do_set_related_action( VMARG )
{
    Item* i_act = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || i_act->isNil() || !i_act->isObject()
        || !IS_DERIVED( i_act, GtkAction ) )
        throw_inv_params( "GtkAction" );
#endif
    GtkAction* act = (GtkAction*) COREGOBJECT( i_act )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_activatable_do_set_related_action( (GtkActivatable*)_obj, act );
}


/*#
    @method get_related_action GtkActivatable
    @brief Gets the related GtkAction for activatable.
    @return the related GtkAction if one is set.
 */
FALCON_FUNC Activatable::get_related_action( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkAction* act = gtk_activatable_get_related_action( (GtkActivatable*)_obj );
    if ( act )
        vm->retval( new Gtk::Action( vm->findWKI( "GtkAction" )->asClass(), act ) );
    else
        vm->retnil();
}


/*#
    @method get_use_action_appearance GtkActivatable
    @brief Gets whether this activatable should reset its layout and appearance when setting the related action or when the action changes appearance.
    @return whether activatable uses its actions appearance.
 */
FALCON_FUNC Activatable::get_use_action_appearance( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_activatable_get_use_action_appearance( (GtkActivatable*)_obj ) );
}


/*#
    @method sync_action_properties GtkActivatable
    @brief This is called to update the activatable completely, this is called internally when the "related-action" property is set or unset and by the implementing class when "use-action-appearance" changes.
    @param action the related GtkAction or NULL.
 */
FALCON_FUNC Activatable::sync_action_properties( VMARG )
{
    Item* i_act = vm->param( 0 );
    // this methods accepts nil
#ifndef NO_PARAMETER_CHECK
    if ( i_act && ( i_act->isNil() || !i_act->isObject()
        || !IS_DERIVED( i_act, GtkAction ) ) )
        throw_inv_params( "GtkAction" );
#endif
    GtkAction* act = NULL;
    if ( i_act )
        act = (GtkAction*) COREGOBJECT( i_act )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_activatable_sync_action_properties( (GtkActivatable*)_obj, act );
}


/*#
    @method set_related_action GtkActivatable
    @brief Sets the related action on the activatable object.
    @param action the GtkAction to set
 */
FALCON_FUNC Activatable::set_related_action( VMARG )
{
    Item* i_act = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || i_act->isNil() || !i_act->isObject()
        || !IS_DERIVED( i_act, GtkAction ) )
        throw_inv_params( "GtkAction" );
#endif
    GtkAction* act = (GtkAction*) COREGOBJECT( i_act )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_activatable_set_related_action( (GtkActivatable*)_obj, act );
}


/*#
    @method set_use_action_appearance GtkActivatable
    @brief Sets whether this activatable should reset its layout and appearance when setting the related action or when the action changes appearance
    @param use_appearance whether to use the actions appearance
 */
FALCON_FUNC Activatable::set_use_action_appearance( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_activatable_set_use_action_appearance(
        (GtkActivatable*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


} // Gtk
} // Falcon

#endif // GTK_MINOR_VERSION >= 16
