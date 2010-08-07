/**
 *  \file gtk_Dialog.cpp
 */

#include "gtk_Dialog.hpp"

#include "gtk_Widget.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void Dialog::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_Dialog = mod->addClass( "GtkDialog", &Dialog::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkWindow" ) );
    c_Dialog->getClassDef()->addInheritance( in );

    c_Dialog->setWKS( true );
    c_Dialog->getClassDef()->factory( &Dialog::factory );

    Gtk::MethodTab methods[] =
    {
    //{ "new_with_buttons",       &Dialog::new_with_buttons },
    { "run",                    &Dialog::run },
    { "response",               &Dialog::response },
    { "add_button",             &Dialog::add_button },
    //{ "add_buttons",            &Dialog::add_buttons },
    { "add_action_widget",      &Dialog::add_action_widget },
    { "get_has_separator",      &Dialog::get_has_separator },
    { "set_default_response",   &Dialog::set_default_response },
    { "set_has_separator",      &Dialog::set_has_separator },
    { "set_response_sensitive", &Dialog::set_response_sensitive },
#if GTK_CHECK_VERSION( 2, 8, 0 )
    { "get_response_for_widget",&Dialog::get_response_for_widget },
#endif
#if GTK_CHECK_VERSION( 2, 20, 0 )
    { "get_widget_for_response",&Dialog::get_widget_for_response },
#endif
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_action_area",        &Dialog::get_action_area },
    { "get_content_area",       &Dialog::get_content_area },
#endif
#if GTK_CHECK_VERSION( 2, 6, 0 )
    //{ "alternative_dialog_button_order",&Dialog::alternative_dialog_button_order },
    //{ "set_alternative_button_order",&Dialog::set_alternative_button_order },
    //{ "set_alternative_button_order_from_array",&Dialog::set_alternative_button_order_from_array },
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_Dialog, meth->name, meth->cb );
}


Dialog::Dialog( const Falcon::CoreClass* gen, const GtkDialog* dlg )
    :
    Gtk::CoreGObject( gen, (GObject*) dlg )
{}


Falcon::CoreObject* Dialog::factory( const Falcon::CoreClass* gen, void* dlg, bool )
{
    return new Dialog( gen, (GtkDialog*) dlg );
}


/*#
    @class GtkDialog
    @brief Create popup windows

    Dialog boxes are a convenient way to prompt the user for a small amount of input,
    e.g. to display a message, ask a question, or anything else that does not require
    extensive effort on the user's part.

    GTK+ treats a dialog as a window split vertically. The top section is a GtkVBox,
    and is where widgets such as a GtkLabel or a GtkEntry should be packed. The bottom
    area is known as the action_area. This is generally used for packing buttons into
    the dialog which may perform functions such as cancel, ok, or apply. The two areas
    are separated by a GtkHSeparator.

    GtkDialog boxes are created with their constructor or new_with_buttons().
    new_with_buttons() is recommended; it allows you to set the dialog title,
    some convenient flags, and add simple buttons.

    If 'dialog' is a newly created dialog, the two primary areas of the window can be
    accessed through get_content_area() and get_action_area(),
    as can be seen from the example, below.

    A 'modal' dialog (that is, one which freezes the rest of the application from
    user input), can be created by calling set_modal() on the dialog. Use
    the GTK_WINDOW() macro to cast the widget returned from new() into a
    GtkWindow. When using new_with_buttons() you can also pass the GTK_DIALOG_MODAL
    flag to make a dialog modal.

    If you add buttons to GtkDialog using new_with_buttons(), add_button(),
    add_buttons(), or add_action_widget(), clicking the button will
    emit a signal called "response" with a response ID that you specified. GTK+ will never
    assign a meaning to positive response IDs; these are entirely user-defined.
    But for convenience, you can use the response IDs in the GtkResponseType enumeration
    (these all have values less than zero). If a dialog receives a delete event, the
    "response" signal will be emitted with a response ID of GTK_RESPONSE_DELETE_EVENT.

    If you want to block waiting for a dialog to return before returning control flow
    to your code, you can call run(). This function enters a recursive main
    loop and waits for the user to respond to the dialog, returning the response ID
    corresponding to the button the user clicked.

    For the simple dialog in the following example, in reality you'd probably use
    GtkMessageDialog to save yourself some effort. But you'd need to create the
    dialog contents manually if you had more than a simple message in the dialog.

    [...]

 */
FALCON_FUNC Dialog::init( VMARG )
{
    MYSELF;

    if ( self->getObject() )
        return;

#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_dialog_new();
    self->setObject( (GObject*) wdt );
}


//FALCON_FUNC Dialog::new_with_buttons( VMARG );


/*#
    @method run GtkDialog
    @brief To be completed..
    @return response ID
 */
FALCON_FUNC Dialog::run( VMARG )
{
#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_dialog_run( (GtkDialog*)_obj ) );
}


/*#
    @method response GtkDialog
    @brief Emits the "response" signal with the given response ID.
    @param response_id response ID

    Used to indicate that the user has responded to the dialog in some way; typically
    either you or gtk_dialog_run() will be monitoring the ::response signal and take
    appropriate action
 */
FALCON_FUNC Dialog::response( VMARG )
{
    Item* i_res = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_res || i_res->isNil() || !i_res->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_dialog_response( (GtkDialog*)_obj, i_res->asInteger() );
}


/*#
    @method add_button GtkDialog
    @brief Adds a button with the given text (or a stock button, if button_text is a stock ID) and sets things up so that clicking the button will emit the "response" signal with the given response_id.
    @param button_text text of button, or stock ID
    @param response_id response ID for the button
    @return the button widget that was added (GtkWidget)

    The button is appended to the end of the dialog's action area. The button widget
    is returned, but usually you don't need it.
 */
FALCON_FUNC Dialog::add_button( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,I" );

    char* txt = args.getCString( 0 );
    int id = args.getInteger( 1 );

    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_dialog_add_button( (GtkDialog*)_obj, txt, id );
    vm->retval( new Gtk::Widget( vm->findWKI( "Button" )->asClass(), wdt ) );
}


//FALCON_FUNC Dialog::add_buttons( VMARG );


/*#
    @method add_action_widget
    @brief Adds an activatable widget to the action area of a GtkDialog, connecting a signal handler that will emit the "response" signal on the dialog when the widget is activated.
    @param child an activatable widget
    @param response_id response ID for child

    The widget is appended to the end of the dialog's action area. If you want
    to add a non-activatable widget, simply pack it into the action_area field of
    the GtkDialog struct.
 */
FALCON_FUNC Dialog::add_action_widget( VMARG )
{
    Item* i_child = vm->param( 0 );
    Item* i_id = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_child || i_child->isNil() || !i_child->isObject()
        || !IS_DERIVED( i_child, GtkWidget )
        || !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "GtkWidget,I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* child = (GtkWidget*) COREGOBJECT( i_child )->getObject();
    gtk_dialog_add_action_widget( (GtkDialog*)_obj, child, i_id->asInteger() );
}


/*#
    @method get_has_separator
    @brief Accessor for whether the dialog has a separator.
    @return true if the dialog has a separator.
 */
FALCON_FUNC Dialog::get_has_separator( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_dialog_get_has_separator( (GtkDialog*)_obj ) );
}


/*#
    @method set_default_response
    @brief Sets the last widget in the dialog's action area with the given response_id as the default widget for the dialog.
    @param response_id a response ID

    Pressing "Enter" normally activates the default widget.
 */
FALCON_FUNC Dialog::set_default_response( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_dialog_set_default_response( (GtkDialog*)_obj, i_id->asInteger() );
}

/*#
    @method set_has_separator
    @brief Sets whether the dialog has a separator above the buttons. TRUE by default.
    @param setting true to have a separator
 */
FALCON_FUNC Dialog::set_has_separator( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_dialog_set_has_separator( (GtkDialog*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method set_response_sensitive
    @brief Calls gtk_widget_set_sensitive (widget, setting) for each widget in the dialog's action area with the given response_id.
    @param response_id a response ID
    @param setting true for sensitive

    A convenient way to sensitize/desensitize dialog buttons.
 */
FALCON_FUNC Dialog::set_response_sensitive( VMARG )
{
    Item* i_id = vm->param( 0 );
    Item* i_bool = vm->param( 1 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger()
        || !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "I,B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_dialog_set_response_sensitive( (GtkDialog*)_obj, i_id->asInteger(),
            i_bool->asBoolean() ? TRUE : FALSE );
}


#if GTK_CHECK_VERSION( 2, 8, 0 )
/*#
    @method get_response_for_widget
    @brief Gets the response id of a widget in the action area of a dialog.
    @param widget a widget in the action area of dialog
    @return the response id of widget, or GTK_RESPONSE_NONE if widget doesn't have a response id set.
 */
FALCON_FUNC Dialog::get_response_for_widget( VMARG )
{
    Item* i_wdt = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_wdt || i_wdt->isNil() || !i_wdt->isObject()
        || !IS_DERIVED( i_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = (GtkWidget*) COREGOBJECT( i_wdt )->getObject();
    vm->retval( gtk_dialog_get_response_for_widget( (GtkDialog*)_obj, wdt ) );
}
#endif


#if GTK_CHECK_VERSION( 2, 20, 0 )
/*#
    @method get_widget_for_response
    @brief Gets the widget button that uses the given response ID in the action area of a dialog.
    @param response_id the response ID used by the dialog widget
    @return the widget button that uses the given response_id, or nil.
 */
FALCON_FUNC Dialog::get_widget_for_response( VMARG )
{
    Item* i_id = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_id || i_id->isNil() || !i_id->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_dialog_get_widget_for_response( (GtkDialog*)_obj,
            i_id->asInteger() );
    if ( !wdt )
        vm->retnil();
    else
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}
#endif


#if GTK_CHECK_VERSION( 2, 14, 0 )
/*#
    @method get_action_area
    @brief Returns the action area of dialog.
    @return the action area.
 */
FALCON_FUNC Dialog::get_action_area( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_dialog_get_action_area( (GtkDialog*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}


/*#
    @method get_content_area
    @brief Returns the content area of dialog.
    @return the content area.
 */
FALCON_FUNC Dialog::get_content_area( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_dialog_get_action_area( (GtkDialog*)_obj );
    vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
}
#endif // GTK_CHECK_VERSION( 2, 14, 0 )


#if GTK_CHECK_VERSION( 2, 6, 0 )
//FALCON_FUNC Dialog::alternative_dialog_button_order( VMARG );

//FALCON_FUNC Dialog::set_alternative_button_order( VMARG );

//FALCON_FUNC Dialog::set_alternative_button_order_from_array( VMARG );
#endif // GTK_CHECK_VERSION( 2, 6, 0 )


} // Gtk
} // Falcon
