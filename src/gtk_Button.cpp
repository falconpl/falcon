/**
 *  \file gtk_Button.cpp
 */

#include "gtk_Button.hpp"

#include "gtk_Activatable.hpp"
#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Button::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Button = mod->addClass( "GtkButton", &Button::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Button->getClassDef()->addInheritance( in );

    c_Button->setWKS( true );
    c_Button->getClassDef()->factory( &Button::factory );

    Gtk::MethodTab methods[] =
    {
    { "signal_activate",    &Button::signal_activate },
    { "signal_clicked",     &Button::signal_clicked },
    { "signal_enter",       &Button::signal_enter },
    { "signal_leave",       &Button::signal_leave },
    { "signal_pressed",     &Button::signal_pressed },
    { "signal_released",    &Button::signal_released },
    { "new_with_label",     &Button::new_with_label },
    { "new_with_mnemonic",  &Button::new_with_mnemonic },
    { "new_from_stock",     &Button::new_from_stock },
    { "pressed",            &Button::pressed },
    { "released",           &Button::pressed },
    { "clicked",            &Button::clicked },
    { "enter",              &Button::enter },
    { "leave",              &Button::leave },
    { "set_relief",         &Button::set_relief },
    { "get_relief",         &Button::get_relief },
    { "set_label",          &Button::set_label },
    { "get_label",          &Button::get_label },
    { "set_use_stock",      &Button::set_use_stock },
    { "get_use_stock",      &Button::get_use_stock },
    { "set_use_underline",  &Button::set_use_underline },
    { "get_use_underline",  &Button::get_use_underline },
    { "set_focus_on_click", &Button::set_focus_on_click },
    { "get_focus_on_click", &Button::get_focus_on_click },
    { "set_alignment",      &Button::set_alignment },
    { "get_alignment",      &Button::get_alignment },
    { "set_image",          &Button::set_image },
    { "get_image",          &Button::get_image },
    { "set_image_position", &Button::set_image_position },
    { "get_image_position", &Button::get_image_position },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Button, meth->name, meth->cb );

#if GTK_MINOR_VERSION >= 16
    Gtk::Activatable::clsInit( mod, c_Button );
#endif

}


Button::Button( const Falcon::CoreClass* gen, const GtkButton* btn )
    :
    Gtk::CoreGObject( gen, (GObject*) btn )
{}


Falcon::CoreObject* Button::factory( const Falcon::CoreClass* gen, void* btn, bool )
{
    return new Button( gen, (GtkButton*) btn );
}


/*#
    @class GtkButton
    @brief A widget that creates a signal when clicked on

    The GtkButton widget is generally used to attach a function to that is called
    when the button is pressed. The various signals and how to use them are
    outlined below.

    The GtkButton widget can hold any valid child widget. That is it can hold
    most any other standard GtkWidget. The most commonly used child is the GtkLabel.
 */
FALCON_FUNC Button::init( VMARG )
{
    MYSELF;
    if ( self->getGObject() )
        return;
    NO_ARGS
    self->setGObject( (GObject*) gtk_button_new() );
}


/*#
    @method signal_activate GtkButton
    @brief The activate signal on GtkButton is an action signal and emitting it causes the button to animate press then release.

    Applications should never connect to this signal, but use the "clicked" signal.
 */
FALCON_FUNC Button::signal_activate( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "activate", (void*) &Button::on_activate, vm );
}


void Button::on_activate( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method signal_clicked GtkButton
    @brief Emitted when the button has been activated (pressed and released).
 */
FALCON_FUNC Button::signal_clicked( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "clicked", (void*) &Button::on_clicked, vm );
}


void Button::on_clicked( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "clicked", "on_clicked", (VMachine*)_vm );
}


/*#
    @method signal_enter GtkButton
    @brief Emitted when the pointer enters the button.

    GtkButton::enter has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the "enter-notify-event" signal.
 */
FALCON_FUNC Button::signal_enter( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "enter", (void*) &Button::on_enter, vm );
}


void Button::on_enter( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "enter", "on_enter", (VMachine*)_vm );
}


/*#
    @method signal_leave GtkButton
    @brief Emitted when the pointer leaves the button.

    GtkButton::leave has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the Widget "leave-notify-event" signal.
 */
FALCON_FUNC Button::signal_leave( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "leave", (void*) &Button::on_leave, vm );
}


void Button::on_leave( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "leave", "on_leave", (VMachine*)_vm );
}


/*#
    @method signal_pressed GtkButton
    @brief Emitted when the button is pressed.

    GtkButton::pressed has been deprecated since version 2.8 and should not be used in
    newly-written code. Use the Widget "button-press-event" signal.
 */
FALCON_FUNC Button::signal_pressed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "pressed", (void*) &Button::on_pressed, vm );
}


void Button::on_pressed( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "pressed", "on_pressed", (VMachine*)_vm );
}


/*#
    @method signal_released GtkButton
    @brief Emitted when the button is released.

    GtkButton::released has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the Widget "button-release-event" signal.
 */
FALCON_FUNC Button::signal_released( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "released", (void*) &Button::on_released, vm );
}


void Button::on_released( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "released", "on_released", (VMachine*)_vm );
}


/*#
    @method new_with_label GtkButton
    @brief Creates a GtkButton widget with a GtkLabel child containing the given text.
    @param label The text you want the GtkLabel to hold.
    @return The newly created GtkButton widget.
 */
FALCON_FUNC Button::new_with_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_button_new_with_label( lbl.c_str() );
    vm->retval( new Gtk::Button( vm->findWKI( "GtkButton" )->asClass(),
                                 (GtkButton*) btn ) );
}


/*#
    @method new_with_mnemonic GtkButton
    @brief Creates a new GtkButton containing a label.
    @param label The text of the button, with an underscore in front of the mnemonic character
    @return a new GtkButton

    If characters in label are preceded by an underscore, they are underlined.
    If you need a literal underscore character in a label, use '__'
    (two underscores). The first underlined character represents a keyboard
    accelerator called a mnemonic. Pressing Alt and that key activates the button.
 */
FALCON_FUNC Button::new_with_mnemonic( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    GtkWidget* btn = gtk_button_new_with_mnemonic( lbl.c_str() );
    vm->retval( new Gtk::Button( vm->findWKI( "GtkButton" )->asClass(),
                                 (GtkButton*) btn ) );
}


/*#
    @method new_from_stock GtkButton
    @brief Creates a new GtkButton containing the image and text from a stock item.
    @param stock_id the name of the stock item
    @return a new GtkButton

    Some stock ids have preprocessor macros like GTK_STOCK_OK and GTK_STOCK_APPLY.

    If stock_id is unknown, then it will be treated as a mnemonic label (as for
    gtk_button_new_with_mnemonic()).
 */
FALCON_FUNC Button::new_from_stock( VMARG )
{
    Item* i_stock = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_stock || !i_stock->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString stock( i_stock->asString() );
    GtkWidget* btn = gtk_button_new_from_stock( stock.c_str() );
    vm->retval( new Gtk::Button( vm->findWKI( "GtkButton" )->asClass(),
                                 (GtkButton*) btn ) );
}


/*#
    @method pressed GtkButton
    @brief Emits a "pressed" signal to the given GtkButton.
 */
FALCON_FUNC Button::pressed( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_pressed( (GtkButton*)_obj );
}


/*#
    @method released GtkButton
    @brief Emits a "released" signal to the given GtkButton.
 */
FALCON_FUNC Button::released( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_released( (GtkButton*)_obj );
}


/*#
    @method clicked GtkButton
    @brief Emits a "clicked" signal to the given GtkButton.
 */
FALCON_FUNC Button::clicked( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_clicked( (GtkButton*)_obj );
}


/*#
    @method enter GtkButton
    @brief Emits a "enter" signal to the given GtkButton.
 */
FALCON_FUNC Button::enter( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_enter( (GtkButton*)_obj );
}


/*#
    @method leave GtkButton
    @brief Emits a "leave" signal to the given GtkButton.
 */
FALCON_FUNC Button::leave( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_leave( (GtkButton*)_obj );
}


/*#
    @method set_relief GtkButton
    @brief Sets the relief style of the edges of the given GtkButton widget.
    @param newstyle The new GtkReliefStyle.

    Three styles exist, GTK_RELIEF_NORMAL, GTK_RELIEF_HALF, GTK_RELIEF_NONE.
    The default style is, as one can guess, GTK_RELIEF_NORMAL.
 */
FALCON_FUNC Button::set_relief( VMARG )
{
    Item* i_style = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_style || !i_style->isInteger() )
        throw_inv_params( "GtkReliefStyle" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_relief( (GtkButton*)_obj, (GtkReliefStyle) i_style->asInteger() );
}


/*#
    @method get_relief GtkButton
    @brief Returns the current relief style of the given GtkButton.
    @return The current GtkReliefStyle.
 */
FALCON_FUNC Button::get_relief( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_button_get_relief( (GtkButton*)_obj ) );
}


/*#
    @method set_label GtkButton
    @brief Sets the text of the label of the button.
    @param label a string

    This text is also used to select the stock item if gtk_button_set_use_stock() is used.
    This will also clear any previously set labels.
 */
FALCON_FUNC Button::set_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || !i_lbl->isString() )
        throw_inv_params( "S" );
#endif
    AutoCString lbl( i_lbl->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_label( (GtkButton*)_obj, lbl.c_str() );
}


/*#
    @method get_label GtkButton
    @brief Fetches the text from the label of the button.
    @return The text of the label widget, or nil.

    If the label text has not been set the return value will be NULL.
    This will be the case if you create an empty button with gtk_button_new()
    to use as a container.
 */
FALCON_FUNC Button::get_label( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_button_get_label( (GtkButton*)_obj );
    if ( lbl )
        vm->retval( UTF8String( lbl ) );
    else
        vm->retnil();
}


/*#
    @method set_use_stock GtkButton
    @brief Use a stock id.
    @param use_stock TRUE if the button should use a stock item

    If true, the label set on the button is used as a stock id to select the
    stock item for the button.
 */
FALCON_FUNC Button::set_use_stock( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_use_stock( (GtkButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_use_stock GtkButton
    @brief Returns whether the button label is a stock item.
    @return TRUE if the button label is used to select a stock item instead of being used directly as the label text.
 */
FALCON_FUNC Button::get_use_stock( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_use_stock( (GtkButton*)_obj ) );
}


/*#
    @method set_use_underline GtkButton
    @brief Sets an underline.
    @param use_underline TRUE if underlines in the text indicate mnemonics

    If true, an underline in the text of the button label indicates the next
    character should be used for the mnemonic accelerator key.
 */
FALCON_FUNC Button::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_use_underline( (GtkButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_use_underline GtkButton
    @brief Returns whether an embedded underline in the button label indicates a mnemonic.
    @return TRUE if an embedded underline in the button label indicates the mnemonic accelerator keys.
 */
FALCON_FUNC Button::get_use_underline( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_use_underline( (GtkButton*)_obj ) );
}


/*#
    @method set_focus_on_click GtkButton
    @brief Sets whether the button will grab focus when it is clicked with the mouse.
    @param focus_on_click whether the button grabs focus when clicked with the mouse

    Making mouse clicks not grab focus is useful in places like toolbars where
    you don't want the keyboard focus removed from the main area of the application.
 */
FALCON_FUNC Button::set_focus_on_click( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_focus_on_click( (GtkButton*)_obj, (gboolean) i_bool->asBoolean() );
}


/*#
    @method get_focus_on_click GtkButton
    @brief Returns whether the button grabs focus when it is clicked with the mouse.
    @return TRUE if the button grabs focus when it is clicked with the mouse.
 */
FALCON_FUNC Button::get_focus_on_click( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_focus_on_click( (GtkButton*)_obj ) );
}


/*#
    @method set_alignment GtkButton
    @brief Sets the alignment of the child.
    @param xalign the horizontal position of the child, 0.0 is left aligned, 1.0 is right aligned
    @param yalign the vertical position of the child, 0.0 is top aligned, 1.0 is bottom aligned

    This property has no effect unless the child is a GtkMisc or a GtkAligment.
 */
FALCON_FUNC Button::set_alignment( VMARG )
{
    Item* i_x = vm->param( 0 );
    Item* i_y = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_x || !i_x->isOrdinal()
        || !i_y || !i_y->isOrdinal() )
        throw_inv_params( "N,N" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_alignment( (GtkButton*)_obj, i_x->forceNumeric(), i_y->forceNumeric() );
}


/*#
    @method get_alignment GtkButton
    @brief Gets the alignment of the child in the button.
    @return an array [ horizontal alignment, vertical alignment ]
 */
FALCON_FUNC Button::get_alignment( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    gfloat x, y;
    gtk_button_get_alignment( (GtkButton*)_obj, &x, &y );
    CoreArray* arr = new CoreArray( 2 );
    arr->append( (numeric) x );
    arr->append( (numeric) y );
    vm->retval( arr );
}


/*#
    @method set_image GtkButton
    @brief Set the image of button to the given widget.
    @param image a widget to set as the image for the button.

    Note that it depends on the "gtk-button-images" setting whether the image will
    be displayed or not, you don't have to call gtk_widget_show() on image yourself.
 */
FALCON_FUNC Button::set_image( VMARG )
{
    Item* i_img = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_img || !( i_img->isObject() && IS_DERIVED( i_img, GtkWidget ) ) )
        throw_inv_params( "GtkWidget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* img = (GtkWidget*) COREGOBJECT( i_img )->getGObject();
    gtk_button_set_image( (GtkButton*)_obj, img );
}


/*#
    @method get_image GtkButton
    @brief Gets the widget that is currenty set as the image of button.
    @return a GtkWidget or nil in case there is no image

    This may have been explicitly set by gtk_button_set_image() or constructed by
    gtk_button_new_from_stock().
 */
FALCON_FUNC Button::get_image( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* img = gtk_button_get_image( (GtkButton*)_obj );
    if ( img )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), img ) );
    else
        vm->retnil();
}


/*#
    @method set_image_position GtkButton
    @brief Sets the position of the image relative to the text inside the button.
    @param position the position (GtkPositionType).
 */
FALCON_FUNC Button::set_image_position( VMARG )
{
    Item* i_pos = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_pos || !i_pos->isInteger() )
        throw_inv_params( "GtkPositionType" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_image_position( (GtkButton*)_obj, (GtkPositionType) i_pos->asInteger() );
}


/*#
    @method get_image_position GtkButton
    @brief Gets the position of the image relative to the text inside the button.
    @return the position (GtkPositionType).
 */
FALCON_FUNC Button::get_image_position( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_button_get_image_position( (GtkButton*)_obj ) );
}


} // Gtk
} // Falcon
