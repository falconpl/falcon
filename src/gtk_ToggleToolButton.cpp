/**
 *  \file gtk_ToggleToolButton.cpp
 */

#include "gtk_ToggleToolButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToggleToolButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToggleToolButton = mod->addClass( "GtkToggleToolButton", &ToggleToolButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToolButton" ) );
    c_ToggleToolButton->getClassDef()->addInheritance( in );

    c_ToggleToolButton->setWKS( true );
    c_ToggleToolButton->getClassDef()->factory( &ToggleToolButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_toggled",     &ToggleToolButton::signal_toggled },
    { "new_from_stock",     &ToggleToolButton::new_from_stock },
    { "set_active",         &ToggleToolButton::set_active },
    { "get_active",         &ToggleToolButton::get_active },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToggleToolButton, meth->name, meth->cb );
}


ToggleToolButton::ToggleToolButton( const Falcon::CoreClass* gen, const GtkToggleToolButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ToggleToolButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ToggleToolButton( gen, (GtkToggleToolButton*) btn );
}


/*#
    @class GtkToggleToolButton
    @brief A GtkToolItem containing a toggle button

    A GtkToggleToolButton is a GtkToolItem that contains a toggle button.
 */
FALCON_FUNC ToggleToolButton::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;

#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    self->setGObject( (GObject*) gtk_toggle_tool_button_new() );
}


/*#
    @method signal_toggled GtkToggleToolButton
    @brief Emitted whenever the toggle tool button changes state.
 */
FALCON_FUNC ToggleToolButton::signal_toggled( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "toggled", (void*) &ToggleToolButton::on_toggled, vm );
}


void ToggleToolButton::on_toggled( GtkToggleToolButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "toggled", "on_toggled", (VMachine*)_vm );
}


/*#
    @method new_from_stock GtkToggleToolButton
    @brief Creates a new GtkToggleToolButton containing the image and text from a stock item.
    @param stock_id the name of the stock item
    @return A new GtkToggleToolButton

    Some stock ids have preprocessor macros like GTK_STOCK_OK and GTK_STOCK_APPLY.

    It is an error if stock_id is not a name of a stock item.
 */
FALCON_FUNC ToggleToolButton::new_from_stock( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* stock = args.getCString( 0 );
    GtkToggleToolButton* btn = (GtkToggleToolButton*) gtk_toggle_tool_button_new_from_stock( stock );
    vm->retval( new Gtk::ToggleToolButton( vm->findWKI( "GtkToggleToolButton" )->asClass(), btn ) );
}


/*#
    @method set_active GtkToggleToolButton
    @brief Sets the status of the toggle tool button.
    @param is_active whether button should be active

    Set to TRUE if you want the GtkToggleButton to be 'pressed in', and FALSE to
    raise it. This action causes the toggled signal to be emitted.
 */
FALCON_FUNC ToggleToolButton::set_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_toggle_tool_button_set_active( (GtkToggleToolButton*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_active GtkToggleToolButton
    @brief Queries a GtkToggleToolButton and returns its current state.
    @return TRUE if the toggle tool button is pressed in, FALSE if not

    Returns TRUE if the toggle button is pressed in and FALSE if it is raised.
 */
FALCON_FUNC ToggleToolButton::get_active( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_toggle_tool_button_get_active( (GtkToggleToolButton*)_obj ) );
}


} // Gtk
} // Falcon
