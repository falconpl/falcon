/**
 *  \file gtk_Button.cpp
 */

#include "gtk_Button.hpp"

#include "gtk_Activatable.hpp"
#include "gtk_Widget.hpp"

#include <gtk/gtk.h>


namespace Falcon {
namespace Gtk {

void Button::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Button = mod->addClass( "GtkButton", &Button::init )
        ->addParam( "label" )
        ->addParam( "mode" );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkBin" ) );
    c_Button->getClassDef()->addInheritance( in );

    c_Button->getClassDef()->factory( &Button::factory );

    mod->addClassProperty( c_Button, "NO_MNEMONIC" ).setInteger( 0 ).setReadOnly( true );
    mod->addClassProperty( c_Button, "MNEMONIC" ).setInteger( 1 ).setReadOnly( true );
    mod->addClassProperty( c_Button, "STOCK" ).setInteger( 2 ).setReadOnly( true );


    Gtk::MethodTab methods[] =
    {
    //{ "signal_activate",    &Button::signal_activate },
    { "signal_activate",    &Button::signal_activate },
    { "signal_clicked",     &Button::signal_clicked },
    { "signal_enter",       &Button::signal_enter },
    { "signal_leave",       &Button::signal_leave },
    { "signal_pressed",     &Button::signal_pressed },
    { "signal_released",    &Button::signal_released },
    { "pressed",            &Button::pressed },
    { "released",           &Button::pressed },
    { "clicked",            &Button::clicked },
    { "enter",              &Button::enter },
    { "leave",              &Button::leave },
    //{ "set_relief",         &Button::set_relief },
    //{ "get_relief",         &Button::get_relief },
    { "set_label",          &Button::set_label },
    { "get_label",          &Button::get_label },
    { "set_use_stock",      &Button::set_use_stock },
    { "get_use_stock",      &Button::get_use_stock },
    { "set_use_underline",  &Button::set_use_underline },
    { "get_use_underline",  &Button::get_use_underline },
    { "set_focus_on_click", &Button::set_focus_on_click },
    { "get_focus_on_click", &Button::get_focus_on_click },
    //{ "set_alignment",      &Button::set_alignment },
    //{ "get_alignment",      &Button::get_alignment },
    { "set_image",          &Button::set_image },
    { "get_image",          &Button::get_image },
    //{ "set_image_position", &Button::set_image_position },
    //{ "get_image_position", &Button::get_image_position },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Button, meth->name, meth->cb );

    /*
     *  implements GtkActivatable
     */
    Gtk::Activatable::clsInit( mod, c_Button );
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
    @brief A push button
    @optparam label A label string, or a gtk stock id (string)
    @optparam mode (integer) GtkButton.NO_MNEMONIC (default), or GtkButton.MNEMONIC, or GtkButton.STOCK
    @raise ParamError Invalid argument

    If no arguments are given, creates an empty button.
 */
FALCON_FUNC Button::init( VMARG )
{
    MYSELF;

    if ( self->getGObject() )
        return;

    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( i_lbl && ( i_lbl->isNil() || !i_lbl->isString() ) )
    {
        throw_inv_params( "[S[,I]]" );
    }
#endif
    if ( !i_lbl )
    {
        GtkWidget* btn = gtk_button_new();
        self->setGObject( (GObject*) btn );
        return;
    }

    AutoCString lbl( i_lbl->asString() );
    int mode = 0;

    Item* i_mode = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( i_mode && ( i_mode->isNil() || !i_mode->isInteger() ) )
    {
        throw_inv_params( "[S[,I]]" );
    }
    else
#endif
    if ( i_mode )
    {
        mode = i_mode->asInteger();
        if ( mode < 0 || mode > 2 )
        {
            throw_inv_params( "[S[,I]]" );
        }
    }

    GtkWidget* btn;
    switch ( mode )
    {
    case 0:
        btn = gtk_button_new_with_label( lbl.c_str() );
        break;
    case 1:
        btn = gtk_button_new_with_mnemonic( lbl.c_str() );
        break;
    case 2:
        btn = gtk_button_new_from_stock( lbl.c_str() );
        break;
    default:
        return; // not reached
    }

    self->setGObject( (GObject*) btn );
}


/*#
    @method signal_activate GtkButton
    @brief Connect a VMSlot to the button activate signal and return it

    The activate signal on GtkButton is an action signal and emitting it causes
    the button to animate press then release.
    Applications should never connect to this signal, but use the "clicked" signal.
 */
FALCON_FUNC Button::signal_activate( VMARG )
{
    CoreGObject::get_signal( "activate", (void*) &Button::on_activate, vm );
}


void Button::on_activate( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "activate", "on_activate", (VMachine*)_vm );
}


/*#
    @method signal_clicked GtkButton
    @brief Connect a VMSlot to the button clicked signal and return it

    Emitted when the button has been activated (pressed and released).
 */
FALCON_FUNC Button::signal_clicked( VMARG )
{
    CoreGObject::get_signal( "clicked", (void*) &Button::on_clicked, vm );
}


void Button::on_clicked( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "clicked", "on_clicked", (VMachine*)_vm );
}


/*#
    @method signal_enter GtkButton
    @brief Connect a VMSlot to the button enter signal and return it

    Emitted when the pointer enters the button.

    GtkButton::enter has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the "enter-notify-event" signal.
 */
FALCON_FUNC Button::signal_enter( VMARG )
{
    CoreGObject::get_signal( "enter", (void*) &Button::on_enter, vm );
}


void Button::on_enter( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "enter", "on_enter", (VMachine*)_vm );
}


/*#
    @method signal_leave GtkButton
    @brief Connect a VMSlot to the button leave signal and return it

    Emitted when the pointer leaves the button.

    GtkButton::leave has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the Widget "leave-notify-event" signal.
 */
FALCON_FUNC Button::signal_leave( VMARG )
{
    CoreGObject::get_signal( "leave", (void*) &Button::on_leave, vm );
}


void Button::on_leave( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "leave", "on_leave", (VMachine*)_vm );
}


/*#
    @method signal_pressed GtkButton
    @brief Connect a VMSlot to the button pressed signal and return it

    Emitted when the button is pressed.

    GtkButton::pressed has been deprecated since version 2.8 and should not be used in
    newly-written code. Use the Widget "button-press-event" signal.
 */
FALCON_FUNC Button::signal_pressed( VMARG )
{
    CoreGObject::get_signal( "pressed", (void*) &Button::on_pressed, vm );
}


void Button::on_pressed( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "pressed", "on_pressed", (VMachine*)_vm );
}


/*#
    @method signal_released GtkButton
    @brief Connect a VMSlot to the button released signal and return it

    Emitted when the button is released.

    GtkButton::released has been deprecated since version 2.8 and should not be used
    in newly-written code. Use the Widget "button-release-event" signal.
 */
FALCON_FUNC Button::signal_released( VMARG )
{
    CoreGObject::get_signal( "released", (void*) &Button::on_released, vm );
}


void Button::on_released( GtkButton* btn, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) btn, "released", "on_released", (VMachine*)_vm );
}


/*#
    @method pressed GtkButton
    @brief Emits a "pressed" signal to the given GtkButton.
 */
FALCON_FUNC Button::pressed( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
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
    {
        throw_require_no_args();
    }
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
    {
        throw_require_no_args();
    }
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
    {
        throw_require_no_args();
    }
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
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_leave( (GtkButton*)_obj );
}


//FALCON_FUNC set_relief( VMARG );

//FALCON_FUNC get_relief( VMARG );


/*#
    @method set_label GtkButton
    @brief Sets the text of the label of the button.

    This text is also used to select the stock item if gtk_button_set_use_stock() is used.
    This will also clear any previously set labels.
 */
FALCON_FUNC Button::set_label( VMARG )
{
    Item* i_lbl = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_lbl || i_lbl->isNil() || !i_lbl->isString() )
    {
        throw_inv_params( "S" );
    }
#endif
    AutoCString s( i_lbl->asString() );
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_label( (GtkButton*)_obj, s.c_str() );
}


/*#
    @method get_label GtkButton
    @brief Fetches the text from the label of the button.

    If the label text has not been set the return value will be NULL.
    This will be the case if you create an empty button with gtk_button_new()
    to use as a container.
 */
FALCON_FUNC Button::get_label( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    const gchar* lbl = gtk_button_get_label( (GtkButton*)_obj );
    vm->retval( lbl ? String( lbl ) : String() );
}


/*#
    @method set_use_stock GtkButton
    @brief Use a stock id.
    If true, the label set on the button is used as a stock id to select the
    stock item for the button.
 */
FALCON_FUNC Button::set_use_stock( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_use_stock( (GtkButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_stock GtkButton
    @brief Returns whether the button label is a stock item.
 */
FALCON_FUNC Button::get_use_stock( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_use_stock( (GtkButton*)_obj ) );
}


/*#
    @method set_use_underline GtkButton
    @brief Sets an underline.
    If true, an underline in the text of the button label indicates the next
    character should be used for the mnemonic accelerator key.
 */
FALCON_FUNC Button::set_use_underline( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_use_underline( (GtkButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_underline GtkButton
    @brief Returns whether an embedded underline in the button label indicates a mnemonic.
 */
FALCON_FUNC Button::get_use_underline( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_use_underline( (GtkButton*)_obj ) );
}


/*#
    @method set_focus_on_click GtkButton
    @brief Sets whether the button will grab focus when it is clicked with the mouse.

    Making mouse clicks not grab focus is useful in places like toolbars where
    you don't want the keyboard focus removed from the main area of the application.
 */
FALCON_FUNC Button::set_focus_on_click( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
    {
        throw_inv_params( "B" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_button_set_focus_on_click( (GtkButton*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_focus_on_click GtkButton
    @brief Returns whether the button grabs focus when it is clicked with the mouse.
 */
FALCON_FUNC Button::get_focus_on_click( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_button_get_focus_on_click( (GtkButton*)_obj ) );
}


//FALCON_FUNC Button::set_alignment( VMARG );

//FALCON_FUNC Button::get_alignment( VMARG );


/*#
    @method set_image GtkButton
    @brief Set the image of button to the given widget.

    Note that it depends on the "gtk-button-images" setting whether the image will
    be displayed or not, you don't have to call gtk_widget_show() on image yourself.
 */
FALCON_FUNC Button::set_image( VMARG )
{
    Item* i_img = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_img || i_img->isNil() ||
        !IS_DERIVED( i_img, GtkWidget ) )
    {
        throw_inv_params( "GtkWidget" );
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* img = (GtkWidget*) COREGOBJECT( i_img )->getGObject();
    gtk_button_set_image( (GtkButton*)_obj, img );
}


/*#
    @method get_image GtkButton
    @brief Gets the widget that is currenty set as the image of button.

    This may have been explicitly set by gtk_button_set_image() or constructed by
    gtk_button_new_from_stock().
 */
FALCON_FUNC Button::get_image( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
    {
        throw_require_no_args();
    }
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* gwdt = gtk_button_get_image( (GtkButton*)_obj );
    if ( gwdt )
    {
        Item* wki = vm->findWKI( "GtkWidget" );
        vm->retval( new Gtk::Widget( wki->asClass(), gwdt ) );
    }
    else
        vm->retnil();
}


//FALCON_FUNC Button::set_image_position( VMARG );

//FALCON_FUNC Button::get_image_position( VMARG );



} // Gtk
} // Falcon
