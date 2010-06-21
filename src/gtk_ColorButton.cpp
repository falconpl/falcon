/**
 *  \file gtk_ColorButton.cpp
 */

#include "gtk_ColorButton.hpp"

#include "gdk_Color.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void ColorButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_ColorButton = mod->addClass( "GtkColorButton", &ColorButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_ColorButton->getClassDef()->addInheritance( in );

    c_ColorButton->setWKS( true );
    c_ColorButton->getClassDef()->factory( &ColorButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_color_set",   &ColorButton::signal_color_set },
    { "new_with_color",     &ColorButton::new_with_color },
    { "set_color",          &ColorButton::set_color },
    { "get_color",          &ColorButton::get_color },
    { "set_alpha",          &ColorButton::set_alpha },
    { "get_alpha",          &ColorButton::get_alpha },
    { "set_use_alpha",      &ColorButton::set_use_alpha },
    { "get_use_alpha",      &ColorButton::get_use_alpha },
    { "set_title",          &ColorButton::set_title },
    { "get_title",          &ColorButton::get_title },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_ColorButton, meth->name, meth->cb );
}


ColorButton::ColorButton( const Falcon::CoreClass* gen, const GtkColorButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* ColorButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new ColorButton( gen, (GtkColorButton*) btn );
}


/*#
    @class GtkColorButton
    @brief A button to launch a color selection dialog

    The GtkColorButton is a button which displays the currently selected color
    an allows to open a color selection dialog to change the color. It is
    suitable widget for selecting a color in a preference dialog.

    This returns a widget in the form of a small button containing a swatch
    representing the current selected color. When the button is clicked,
    a color-selection dialog will open, allowing the user to select a color.
    The swatch will be updated to reflect the new color when the user finishes.
 */
FALCON_FUNC ColorButton::init( VMARG )
{
    NO_ARGS
    GtkWidget* wdt = gtk_color_button_new();
    MYSELF;
    self->setGObject( (GObject*) wdt );
}


/*#
    @method signal_color_set GtkColorButton
    @brief The color-set signal is emitted when the user selects a color.

    When handling this signal, use gtk_color_button_get_color() and
    gtk_color_button_get_alpha() to find out which color was just selected.

    Note that this signal is only emitted when the user changes the color.
    If you need to react to programmatic color changes as well, use the notify::color signal.
 */
FALCON_FUNC ColorButton::signal_color_set( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "color_set", (void*) &ColorButton::on_color_set, vm );
}


void ColorButton::on_color_set( GtkColorButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "color_set", "on_color_set", (VMachine*)_vm );
}


/*#
    @method new_with_color GtkColorButton
    @brief Creates a new color button.
    @param color A GdkColor to set the current color with.
    @return a new color button.
 */
FALCON_FUNC ColorButton::new_with_color( VMARG )
{
    Item* i_clr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_clr || i_clr->isNil() || !i_clr->isObject()
        || !IS_DERIVED( i_clr, GdkColor ) )
        throw_inv_params( "GdkColor" );
#endif
    const GdkColor* clr = Falcon::dyncast<Gdk::Color*>( i_clr->asObjectSafe() )->getColor();
    GtkWidget* wdt = gtk_color_button_new_with_color( clr );
    vm->retval( new Gtk::ColorButton( vm->findWKI( "GtkColorButton" )->asClass(),
                                      (GtkColorButton*) wdt ) );
}


/*#
    @method set_color GtkColorButton
    @brief Sets the current color to be color.
    @param color A GdkColor to set the current color with.
 */
FALCON_FUNC ColorButton::set_color( VMARG )
{
    Item* i_clr = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_clr || i_clr->isNil() || !i_clr->isObject()
        || !IS_DERIVED( i_clr, GdkColor ) )
        throw_inv_params( "GdkColor" );
#endif
    const GdkColor* clr = Falcon::dyncast<Gdk::Color*>( i_clr->asObjectSafe() )->getColor();
    MYSELF;
    GET_OBJ( self );
    gtk_color_button_set_color( (GtkColorButton*)_obj, clr );
}


/*#
    @method get_color GtkColorButton
    @brief Gets the current color.
    @return GdkColor of the widget.
 */
FALCON_FUNC ColorButton::get_color( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GdkColor tmp;
    gtk_color_button_get_color( (GtkColorButton*)_obj, &tmp );
    vm->retval( new Gdk::Color( vm->findWKI( "GdkColor" )->asClass(), &tmp ) );
}


/*#
    @method set_alpha GtkColorButton
    @brief Sets the current opacity to be alpha.
    @param alpha an integer between 0 and 65535.
 */
FALCON_FUNC ColorButton::set_alpha( VMARG )
{
    Item* i_a = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_a || i_a->isNil() || !i_a->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_color_button_set_alpha( (GtkColorButton*)_obj, i_a->asInteger() );
}


/*#
    @method get_alpha GtkColorButton
    @brief Returns the current alpha value.
    @return an integer between 0 and 65535.
 */
FALCON_FUNC ColorButton::get_alpha( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_color_button_get_alpha( (GtkColorButton*)_obj ) );
}


/*#
    @method set_use_alpha GtkColorButton
    @brief Sets whether or not the color button should use the alpha channel.
    @param use_alpha true if color button should use alpha channel, false if not.
 */
FALCON_FUNC ColorButton::set_use_alpha( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_color_button_set_use_alpha( (GtkColorButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_use_alpha GtkColorButton
    @brief Does the color selection dialog use the alpha channel?
    @return true if the color sample uses alpha channel, false if not.
 */
FALCON_FUNC ColorButton::get_use_alpha( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_color_button_get_use_alpha( (GtkColorButton*)_obj ) );
}


/*#
    @method set_title GtkColorButton
    @brief Sets the title for the color selection dialog.
    @param title String containing new window title.
 */
FALCON_FUNC ColorButton::set_title( VMARG )
{
    Item* i_title = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_title || !i_title->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString title( i_title->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_color_button_set_title( (GtkColorButton*)_obj, title.c_str() );
}


/*#
    @method get_title GtkColorButton
    @brief Gets the title of the color selection dialog.
    @return The title.
 */
FALCON_FUNC ColorButton::get_title( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const char* title = gtk_color_button_get_title( (GtkColorButton*)_obj );
    vm->retval( UTF8String( title ) );
}


} // Gtk
} // Falcon
