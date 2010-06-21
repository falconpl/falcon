/**
 *  \file gtk_FontButton.cpp
 */

#include "gtk_FontButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void FontButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_FontButton = mod->addClass( "GtkFontButton", &FontButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkButton" ) );
    c_FontButton->getClassDef()->addInheritance( in );

    c_FontButton->setWKS( true );
    c_FontButton->getClassDef()->factory( &FontButton::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_font_set",    &FontButton::signal_font_set },
    { "new_with_font",      &FontButton::new_with_font },
    { "set_font_name",      &FontButton::set_font_name },
    { "get_font_name",      &FontButton::get_font_name },
    { "set_show_style",     &FontButton::set_show_style },
    { "get_show_style",     &FontButton::get_show_style },
    { "set_show_size",      &FontButton::set_show_size },
    { "get_show_size",      &FontButton::get_show_size },
    { "set_use_font",       &FontButton::set_use_font },
    { "get_use_font",       &FontButton::get_use_font },
    { "set_use_size",       &FontButton::set_use_size },
    { "get_use_size",       &FontButton::get_use_size },
    { "set_title",          &FontButton::set_title },
    { "get_title",          &FontButton::get_title },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_FontButton, meth->name, meth->cb );
}


FontButton::FontButton( const Falcon::CoreClass* gen, const GtkFontButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* FontButton::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new FontButton( gen, (GtkFontButton*) btn );
}


/*#
    @class GtkFontButton
    @brief A button to launch a font selection dialog.

    The GtkFontButton is a button which displays the currently selected font an allows
    to open a font selection dialog to change the font. It is suitable widget for
    selecting a font in a preference dialog.
 */
FALCON_FUNC FontButton::init( VMARG )
{
    NO_ARGS
    MYSELF;
    self->setGObject( (GObject*) gtk_font_button_new() );
}


/*#
    @method signal_font_set GtkFontButton
    @brief The font-set signal is emitted when the user selects a font.

    When handling this signal, use get_font_name() to find out
    which font was just selected.

    Note that this signal is only emitted when the user changes the font.
    If you need to react to programmatic font changes as well, use the
    notify::font-name signal.
 */
FALCON_FUNC FontButton::signal_font_set( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "font_set", (void*) &FontButton::on_font_set, vm );
}


void FontButton::on_font_set( GtkFontButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "font_set", "on_font_set", (VMachine*)_vm );
}


/*#
    @method new_with_font GtkFontButton
    @brief Creates a new font picker widget.
    @param fontname Name of font to display in font selection dialog
    @return a new font picker widget.
 */
FALCON_FUNC FontButton::new_with_font( VMARG )
{
    Item* i_fnt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_fnt || !i_fnt->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString fnt( i_fnt->asString() );
    GtkWidget* wdt = gtk_font_button_new_with_font( fnt.c_str() );
    vm->retval( new Gtk::FontButton( vm->findWKI( "GtkFontButton" )->asClass(),
                                     (GtkFontButton*) wdt ) );
}


/*#
    @method set_font_name GtkFontButton
    @brief Sets or updates the currently-displayed font in font picker dialog.
    @param fontname Name of font to display in font selection dialog
    @return Return value of gtk_font_selection_dialog_set_font_name() if the font selection dialog exists, otherwise FALSE.
 */
FALCON_FUNC FontButton::set_font_name( VMARG )
{
    Item* i_fnt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_fnt || !i_fnt->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString fnt( i_fnt->asString() );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_font_button_set_font_name( (GtkFontButton*)_obj,
                                                      fnt.c_str() ) );
}


/*#
    @method get_font_name GtkFontButton
    @brief Retrieves the name of the currently selected font.
    @return the font name.

    This name includes style and size information as well. If you want to render
    something with the font, use this string with pango_font_description_from_string().
    If you're interested in peeking certain values (family name, style, size,
    weight) just query these properties from the PangoFontDescription object.
 */
FALCON_FUNC FontButton::get_font_name( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* nm = gtk_font_button_get_font_name( (GtkFontButton*)_obj );
    vm->retval( UTF8String( nm ) );
}


/*#
    @method set_show_style GtkFontButton
    @brief If show_style is TRUE, the font style will be displayed along with name of the selected font.
    @param show_style TRUE if font style should be displayed in label.
 */
FALCON_FUNC FontButton::set_show_style( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_font_button_set_show_style( (GtkFontButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_show_style GtkFontButton
    @brief Returns whether the name of the font style will be shown in the label.
    @return whether the font style will be shown in the label.
 */
FALCON_FUNC FontButton::get_show_style( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_font_button_get_show_style( (GtkFontButton*)_obj ) );
}


/*#
    @method set_show_size GtkFontButton
    @brief If show_size is TRUE, the font size will be displayed along with the name of the selected font.
    @param show_size TRUE if font size should be displayed in dialog.
 */
FALCON_FUNC FontButton::set_show_size( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_font_button_set_show_size( (GtkFontButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_show_size GtkFontButton
    @brief Returns whether the font size will be shown in the label.
    @return whether the font size will be shown in the label.
 */
FALCON_FUNC FontButton::get_show_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_font_button_get_show_size( (GtkFontButton*)_obj ) );
}


/*#
    @method set_use_font GtkFontButton
    @brief If use_font is TRUE, the font name will be written using the selected font.
    @param use_font If TRUE, font name will be written using font chosen.
 */
FALCON_FUNC FontButton::set_use_font( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_font_button_set_use_font( (GtkFontButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_use_font GtkFontButton
    @brief Returns whether the selected font is used in the label.
    @return whether the selected font is used in the label.
 */
FALCON_FUNC FontButton::get_use_font( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_font_button_get_use_font( (GtkFontButton*)_obj ) );
}


/*#
    @method set_use_size GtkFontButton
    @brief If use_size is TRUE, the font name will be written using the selected size.
    @param use_size If TRUE, font name will be written using the selected size.
 */
FALCON_FUNC FontButton::set_use_size( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_font_button_set_use_size( (GtkFontButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_use_size GtkFontButton
    @brief Returns whether the selected size is used in the label.
    @return whether the selected size is used in the label.
 */
FALCON_FUNC FontButton::get_use_size( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_font_button_get_use_size( (GtkFontButton*)_obj ) );
}


/*#
    @method set_title GtkFontButton
    @brief Sets the title for the font selection dialog.
    @param title a string containing the font selection dialog title
 */
FALCON_FUNC FontButton::set_title( VMARG )
{
    Item* i_title = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_title || !i_title->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString title( i_title->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_font_button_set_title( (GtkFontButton*)_obj, title.c_str() );
}


/*#
    @method get_title GtkFontButton
    @brief Retrieves the title of the font selection dialog.
    @return the title string
 */
FALCON_FUNC FontButton::get_title( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    const gchar* title = gtk_font_button_get_title( (GtkFontButton*)_obj );
    vm->retval( UTF8String( title ) );
}


} // Gtk
} // Falcon
