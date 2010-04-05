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
    //{ "new_with_buttons",       &Dialog:: },
    { "run",                    &Dialog::run },
    { "response",               &Dialog::response },
    { "add_button",             &Dialog::add_button },
#if 0
    { "add_buttons",            &Dialog:: },
    { "add_action_widget",        &Dialog:: },
    { "get_has_separator",        &Dialog:: },
    { "set_default_response",        &Dialog:: },
    { "set_has_separator",        &Dialog:: },
    { "set_response_sensitive",        &Dialog:: },
    { "get_response_for_widget",        &Dialog:: },
    { "get_widget_for_response",        &Dialog:: },
    { "get_action_area",        &Dialog:: },
    { "get_content_area",        &Dialog:: },
    //{ "alternative_dialog_button_order",        &Dialog:: },
    { "set_alternative_button_order",        &Dialog:: },
    { "set_alternative_button_order_from_array",        &Dialog:: },
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

    if ( self->getGObject() )
        return;

#ifndef NO_PARAMETER_CHECK
    if ( vm->paramCount() )
        throw_require_no_args();
#endif
    GtkWidget* wdt = gtk_dialog_new();
    self->setGObject( (GObject*) wdt );
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

#if 0
FALCON_FUNC Dialog::add_buttons( VMARG );

FALCON_FUNC Dialog::add_action_widget( VMARG );

FALCON_FUNC Dialog::get_has_separator( VMARG );

FALCON_FUNC Dialog::set_default_response( VMARG );

FALCON_FUNC Dialog::set_has_separator( VMARG );

FALCON_FUNC Dialog::set_response_sensitive( VMARG );

FALCON_FUNC Dialog::get_response_for_widget( VMARG );

FALCON_FUNC Dialog::get_widget_for_response( VMARG );

FALCON_FUNC Dialog::get_action_area( VMARG );

FALCON_FUNC Dialog::get_content_area( VMARG );

//FALCON_FUNC Dialog::alternative_dialog_button_order( VMARG );

FALCON_FUNC Dialog::set_alternative_button_order( VMARG );

FALCON_FUNC Dialog::set_alternative_button_order_from_array( VMARG );
#endif

} // Gtk
} // Falcon
