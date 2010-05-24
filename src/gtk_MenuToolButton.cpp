/**
 *  \file gtk_MenuToolButton.cpp
 */

#include "gtk_MenuToolButton.hpp"

#include "gtk_Menu.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void MenuToolButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_MenuToolButton = mod->addClass( "GtkMenuToolButton", &MenuToolButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkToolButton" ) );
    c_MenuToolButton->getClassDef()->addInheritance( in );

    c_MenuToolButton->setWKS( true );
    c_MenuToolButton->getClassDef()->factory( &MenuToolButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_show_menu",   &MenuToolButton::signal_show_menu },
    { "new_from_stock",     &MenuToolButton::new_from_stock },
    { "set_menu",           &MenuToolButton::set_menu },
    { "get_menu",           &MenuToolButton::get_menu },
    //{ "set_arrow_tooltip",  &MenuToolButton::set_arrow_tooltip },
    { "set_arrow_tooltip_text",&MenuToolButton::set_arrow_tooltip_text },
    { "set_arrow_tooltip_markup",&MenuToolButton::set_arrow_tooltip_markup },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_MenuToolButton, meth->name, meth->cb );
}


MenuToolButton::MenuToolButton( const Falcon::CoreClass* gen, const GtkMenuToolButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* MenuToolButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new MenuToolButton( gen, (GtkMenuToolButton*) btn );
}


/*#
    @class GtkMenuToolButton
    @brief A GtkToolItem containing a button with an additional dropdown menu
    @param icon_widget a widget that will be used as icon widget, or NULL.
    @param label a string that will be used as label, or NULL.

    A GtkMenuToolButton is a GtkToolItem that contains a button and a small
    additional button with an arrow. When clicked, the arrow button pops up
    a dropdown menu.
 */
FALCON_FUNC MenuToolButton::init( VMARG )
{
    const char* spec = "[GtkWidget,S]";
    Gtk::ArgCheck1 args( vm, spec );
    CoreGObject* o_ico = args.getCoreGObject( 0, false );
    const gchar* lbl = args.getCString( 1, false );
#ifndef NO_PARAMETER_CHECK
    if ( o_ico && !CoreObject_IS_DERIVED( o_ico, GtkWidget ) )
        throw_inv_params( spec );
#endif
    GtkWidget* ico = o_ico ? (GtkWidget*) o_ico->getGObject() : NULL;
    MYSELF;
    self->setGObject( (GObject*) gtk_menu_tool_button_new( ico, lbl ) );
}


/*#
    @method signal_show_menu GtkMenuToolButton
    @brief The show-menu signal is emitted before the menu is shown.

    It can be used to populate the menu on demand, using gtk_menu_tool_button_get_menu().

    Note that even if you populate the menu dynamically in this way, you must
    set an empty menu on the GtkMenuToolButton beforehand, since the arrow is
    made insensitive if the menu is not set.
 */
FALCON_FUNC MenuToolButton::signal_show_menu( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    CoreGObject::get_signal( "show_menu", (void*) &MenuToolButton::on_show_menu, vm );
}


void MenuToolButton::on_show_menu( GtkMenuToolButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "show_menu", "on_show_menu", (VMachine*)_vm );
}


/*#
    @method new_from_stock GtkMenuToolButton
    @brief Creates a new GtkMenuToolButton.
    @param stock_id the name of a stock item
    @return the new GtkMenuToolButton

    The new GtkMenuToolButton will contain an icon and label from the stock item
    indicated by stock_id.
 */
FALCON_FUNC MenuToolButton::new_from_stock( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* stock = args.getCString( 0 );
    GtkMenuToolButton* btn = (GtkMenuToolButton*) gtk_menu_tool_button_new_from_stock( stock );
    vm->retval( new Gtk::MenuToolButton( vm->findWKI( "GtkMenuToolButton" )->asClass(), btn ) );
}


/*#
    @method set_menu GtkMenuToolButton
    @brief Sets the GtkMenu that is popped up when the user clicks on the arrow.
    @param menu the GtkMenu associated with GtkMenuToolButton, or nil.

    If menu is NULL, the arrow button becomes insensitive.
 */
FALCON_FUNC MenuToolButton::set_menu( VMARG )
{
    Item* i_menu = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_menu || !( i_menu->isNil() || ( i_menu->isObject()
        && IS_DERIVED( i_menu, GtkMenu ) ) ) )
        throw_inv_params( "[GtkMenu]" );
#endif
    GtkWidget* menu = i_menu->isNil() ? NULL
                    : (GtkWidget*) COREGOBJECT( i_menu )->getGObject();
    MYSELF;
    GET_OBJ( self );
    gtk_menu_tool_button_set_menu( (GtkMenuToolButton*)_obj, menu );
}


/*#
    @method get_menu GtkMenuToolButton
    @brief Gets the GtkMenu associated with GtkMenuToolButton.
    @return the GtkMenu associated with GtkMenuToolButton
 */
FALCON_FUNC MenuToolButton::get_menu( VMARG )
{
#ifdef STRICT_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* menu = gtk_menu_tool_button_get_menu( (GtkMenuToolButton*)_obj );
    if ( menu )
        vm->retval( new Gtk::Menu( vm->findWKI( "GtkMenu" )->asClass(), (GtkMenu*) menu ) );
    else
        vm->retnil();
}


//FALCON_FUNC MenuToolButton::set_arrow_tooltip( VMARG );


/*#
    @method set_arrow_tooltip_text GtkMenuToolButton
    @brief Sets the tooltip text to be used as tooltip for the arrow button which pops up the menu.
    @param text text to be used as tooltip text for button's arrow button

    See gtk_tool_item_set_tooltip() for setting a tooltip on the whole GtkMenuToolButton.
 */
FALCON_FUNC MenuToolButton::set_arrow_tooltip_text( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* txt = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_tool_button_set_arrow_tooltip_text( (GtkMenuToolButton*)_obj, txt );
}


/*#
    @method set_arrow_tooltip_markup GtkMenuToolButton
    @brief Sets the tooltip markup text to be used as tooltip for the arrow button which pops up the menu.
    @param markup markup text to be used as tooltip text for button's arrow button

    See gtk_tool_item_set_tooltip() for setting a tooltip on the whole GtkMenuToolButton.
 */
FALCON_FUNC MenuToolButton::set_arrow_tooltip_markup( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const gchar* mark = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_menu_tool_button_set_arrow_tooltip_markup( (GtkMenuToolButton*)_obj, mark );
}


} // Gtk
} // Falcon
