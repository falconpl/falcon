/**
 *  \file gtk_ToolButton.cpp
 */

#include "gtk_ToolButton.hpp"

#include "gtk_Widget.hpp"


/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ToolButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ToolButton = mod->addClass( "GtkToolButton", &ToolButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToolItem" ) );
    c_ToolButton->getClassDef()->addInheritance( in );

    c_ToolButton->setWKS( true );
    c_ToolButton->getClassDef()->factory( &ToolButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_clicked",     &ToolButton::signal_clicked },
    { "new_from_stock",     &ToolButton::new_from_stock },
    { "set_label",          &ToolButton::set_label },
    { "get_label",          &ToolButton::get_label },
    { "set_use_underline",  &ToolButton::set_use_underline },
    { "get_use_underline",  &ToolButton::get_use_underline },
    { "set_stock_id",       &ToolButton::set_stock_id },
    { "get_stock_id",       &ToolButton::get_stock_id },
    { "set_icon_name",      &ToolButton::set_icon_name },
    { "get_icon_name",      &ToolButton::get_icon_name },
    { "set_icon_widget",    &ToolButton::set_icon_widget },
    { "get_icon_widget",    &ToolButton::get_icon_widget },
    { "set_label_widget",   &ToolButton::set_label_widget },
    { "get_label_widget",   &ToolButton::get_label_widget },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ToolButton, meth->name, meth->cb );
}


ToolButton::ToolButton( const Falcon::CoreClass* gen, const GtkToolButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ToolButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ToolButton( gen, (GtkToolButton*) btn );
}


/*#
    @class GtkToolButton
    @brief A GtkToolItem subclass that displays buttons
    @param icon_widget a GtkMisc widget that will be used as icon widget, or NULL.
    @param label a string that will be used as label, or NULL.

    GtkToolButtons are GtkToolItems containing buttons.

    The label of a GtkToolButton is determined by the properties "label-widget",
    "label", and "stock-id". If "label-widget" is non-NULL, then that widget is
    used as the label. Otherwise, if "label" is non-NULL, that string is used as
    the label. Otherwise, if "stock-id" is non-NULL, the label is determined by
    the stock item. Otherwise, the button does not have a label.

    The icon of a GtkToolButton is determined by the properties "icon-widget" and
    "stock-id". If "icon-widget" is non-NULL, then that widget is used as the icon.
    Otherwise, if "stock-id" is non-NULL, the icon is determined by the stock item.
    Otherwise, the button does not have a icon.
 */
FALCON_FUNC ToolButton::init( VMARG )
{
    MYSELF;
    if ( self->getObject() )
        return;

    const char* spec = "[GtkWidget,S]";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_ico = args.getCoreGObject( 0, false );
    const gchar* lbl = args.getCString( 1, false );
#ifndef NO_PARAMETER_CHECK
    if ( o_ico && !CoreObject_IS_DERIVED( o_ico, GtkWidget ) )
        throw_inv_params( spec );
#endif
    GtkWidget* ico = o_ico ? (GtkWidget*) o_ico->getObject() : NULL;
    self->setObject( (GObject*) gtk_tool_button_new( ico, lbl ) );
}


/*#
    @method signal_clicked GtkToolButton
    @brief This signal is emitted when the tool button is clicked with the mouse or activated with the keyboard.
 */
FALCON_FUNC ToolButton::signal_clicked( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "clicked", (void*) &ToolButton::on_clicked, vm );
}


void ToolButton::on_clicked( GtkToolButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "clicked", "on_clicked", (VMachine*)_vm );
}


/*#
    @method new_from_stock GtkToolButton
    @brief Creates a new GtkToolButton containing the image and text from a stock item.
    @param stock_id the name of the stock item
    @return A new GtkToolButton

    Some stock ids have preprocessor macros like GTK_STOCK_OK and GTK_STOCK_APPLY.
 */
FALCON_FUNC ToolButton::new_from_stock( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* stock = args.getCString( 0 );
    GtkToolButton* btn = (GtkToolButton*) gtk_tool_button_new_from_stock( stock );
    vm->retval( new Gtk::ToolButton( vm->findWKI( "GtkToolButton" )->asClass(), btn ) );
}


/*#
    @method set_label GtkToolButton
    @brief Sets label as the label used for the tool button.
    @param label a string that will be used as label, or NULL.

    The "label" property only has an effect if not overridden by a non-NULL
    "label_widget" property. If both the "label_widget" and "label" properties
    are NULL, the label is determined by the "stock_id" property. If the
    "stock_id" property is also NULL, button will not have a label.
 */
FALCON_FUNC ToolButton::set_label( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const gchar* lbl = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_label( (GtkToolButton*)_obj, lbl );
}


/*#
    @method get_label GtkToolButton
    @brief Returns the label used by the tool button, or NULL if the tool button doesn't have a label or uses a the label from a stock item.
    @return The label, or NULL
 */
FALCON_FUNC ToolButton::get_label( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_tool_button_get_label( (GtkToolButton*)_obj );
    if ( lbl )
        vm->retval( UTF8String( lbl ) );
    else
        vm->retnil();
}


/*#
    @method set_use_underline GtkToolButton
    @brief If set, an underline in the label property indicates that the next character should be used for the mnemonic accelerator key in the overflow menu
    @param use_underline whether the button label has the form "_Open"

    For example, if the label property is "_Open" and use_underline is TRUE,
    the label on the tool button will be "Open" and the item on the overflow
    menu will have an underlined 'O'.

    Labels shown on tool buttons never have mnemonics on them; this property
    only affects the menu item on the overflow menu.
 */
FALCON_FUNC ToolButton::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_use_underline( (GtkToolButton*)_obj,
                                       i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_underline GtkToolButton
    @brief Returns whether underscores in the label property are used as mnemonics on menu items on the overflow menu.
    @return TRUE if underscores in the label property are used as mnemonics on menu items on the overflow menu.
 */
FALCON_FUNC ToolButton::get_use_underline( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_tool_button_get_use_underline( (GtkToolButton*)_obj ) );
}


/*#
    @method set_stock_id GtkToolButton
    @brief Sets the name of the stock item.
    @param a name of a stock item, or NULL.

    The stock_id property only has an effect if not overridden by non-NULL "label"
    and "icon_widget" properties.
 */
FALCON_FUNC ToolButton::set_stock_id( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const gchar* stock = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_stock_id( (GtkToolButton*)_obj, stock );
}


/*#
    @method get_stock_id GtkToolButton
    @brief Returns the name of the stock item.
    @return the name of the stock item for button.
 */
FALCON_FUNC ToolButton::get_stock_id( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* stock = gtk_tool_button_get_stock_id( (GtkToolButton*)_obj );
    vm->retval( UTF8String( stock ) );
}


/*#
    @method set_icon_name GtkToolButton
    @brief Sets the icon for the tool button from a named themed icon.
    @param the name of the themed icon, or nil.

    The "icon_name" property only has an effect if not overridden by non-NULL
    "label", "icon_widget" and "stock_id" properties.
 */
FALCON_FUNC ToolButton::set_icon_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "[S]" );
    const gchar* ico = args.getCString( 0, false );
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_icon_name( (GtkToolButton*)_obj, ico );
}


/*#
    @method get_icon_name GtkToolButton
    @brief Returns the name of the themed icon for the tool button.
    @return the icon name or NULL if the tool button has no themed icon
 */
FALCON_FUNC ToolButton::get_icon_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* ico = gtk_tool_button_get_icon_name( (GtkToolButton*)_obj );
    if ( ico )
        vm->retval( UTF8String( ico ) );
    else
        vm->retnil();
}


/*#
    @method set_icon_widget GtkToolButton
    @brief Sets icon as the widget used as icon on button.
    @param icon_widget the widget used as icon, or NULL.

    If icon_widget is NULL the icon is determined by the "stock_id" property.
    If the "stock_id" property is also NULL, button will not have an icon.
 */
FALCON_FUNC ToolButton::set_icon_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !( i_wdt->isNil() || ( i_wdt->isObject()
        && IS_DERIVED( i_wdt, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    GtkWidget* wdt = i_wdt->isNil() ? NULL :
                        (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_icon_widget( (GtkToolButton*)_obj, wdt );
}


/*#
    @method get_icon_widget GtkToolButton
    @brief Return the widget used as icon widget on button.
    @return The widget used as icon on button, or NULL.
 */
FALCON_FUNC ToolButton::get_icon_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_tool_button_get_icon_widget( (GtkToolButton*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method set_label_widget GtkToolButton
    @brief Sets label_widget as the widget that will be used as the label for button.
    @param label_widget the widget used as label, or NULL.

    If label_widget is NULL the "label" property is used as label. If "label"
    is also NULL, the label in the stock item determined by the "stock_id"
    property is used as label. If "stock_id" is also NULL, button does not have a label.
 */
FALCON_FUNC ToolButton::set_label_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || !( i_wdt->isNil() || ( i_wdt->isObject()
        && IS_DERIVED( i_wdt, GtkWidget ) ) ) )
        throw_inv_params( "[GtkWidget]" );
#endif
    GtkWidget* wdt = i_wdt->isNil() ? NULL :
                        (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_tool_button_set_label_widget( (GtkToolButton*)_obj, wdt );
}


/*#
    @method get_label_widget GtkToolButton
    @brief Returns the widget used as label on button.
    @return The widget used as label on button, or NULL.
 */
FALCON_FUNC ToolButton::get_label_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_tool_button_get_label_widget( (GtkToolButton*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


} // Gtk
} // Falcon

// vi: set ai et sw=4:
// kate: replace-tabs on; shift-width 4;
