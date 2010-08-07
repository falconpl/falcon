/**
 *  \file gtk_FileChooserButton.cpp
 */

#include "gtk_FileChooserButton.hpp"


namespace Falcon {
namespace Gtk {

/**
 *  \brief module init
 */
void FileChooserButton::modInit( Falcon::Module* mod )
{
    Falcon::Symbol* c_FileChooserButton = mod->addClass(
            "GtkFileChooserButton", &FileChooserButton::init );

    Falcon::InheritDef* in = new Falcon::InheritDef( mod->findGlobalSymbol( "GtkHBox" ) );
    c_FileChooserButton->getClassDef()->addInheritance( in );

    c_FileChooserButton->setWKS( true );
    c_FileChooserButton->getClassDef()->factory( &FileChooserButton::factory );

    Gtk::MethodTab methods[] =
    {
#if GTK_CHECK_VERSION( 2, 12, 0 )
    { "signal_file_set",    &FileChooserButton::signal_file_set },
#endif
    //{ "new_with_backend",   &FileChooserButton::new_with_backend },
    { "new_with_dialog",    &FileChooserButton::new_with_dialog },
    { "get_title",          &FileChooserButton::get_title },
    { "set_title",          &FileChooserButton::set_title },
    { "get_width_chars",    &FileChooserButton::get_width_chars },
    { "set_width_chars",    &FileChooserButton::set_width_chars },
    { "get_focus_on_click", &FileChooserButton::get_focus_on_click },
    { "set_focus_on_click", &FileChooserButton::set_focus_on_click },
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( c_FileChooserButton, meth->name, meth->cb );
}


FileChooserButton::FileChooserButton( const Falcon::CoreClass* gen,
                                      const GtkFileChooserButton* chooser )
    :
    Gtk::CoreGObject( gen, (GObject*) chooser )
{}


Falcon::CoreObject* FileChooserButton::factory( const Falcon::CoreClass* gen, void* chooser, bool )
{
    return new FileChooserButton( gen, (GtkFileChooserButton*) chooser );
}


/*#
    @class GtkFileChooserButton
    @brief A button to launch a file selection dialog
    @param title the title of the browse dialog.
    @param action the open mode for the widget (GtkFileChooserAction).

    The GtkFileChooserButton is a widget that lets the user select a file.
    It implements the GtkFileChooser interface. Visually, it is a file name
    with a button to bring up a GtkFileChooserDialog. The user can then use
    that dialog to change the file associated with that button. This widget
    does not support setting the "select-multiple" property to TRUE.

    [...]
 */
FALCON_FUNC FileChooserButton::init( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S,GtkFileChooserAction" );
    const char* title = args.getCString( 0 );
    int action = args.getInteger( 1 );
    GtkWidget* wdt = gtk_file_chooser_button_new( title, (GtkFileChooserAction) action );
    MYSELF;
    self->setObject( (GObject*) wdt );
}


#if GTK_CHECK_VERSION( 2, 12, 0 )
/*#
    @method signal_file_set GtkFileChooserButton
    @brief The file-set signal is emitted when the user selects a file.

    Note that this signal is only emitted when the user changes the file.
 */
FALCON_FUNC FileChooserButton::signal_file_set( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "file_set", (void*) &FileChooserButton::on_file_set, vm );
}


void FileChooserButton::on_file_set( GtkFileChooserButton* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "file_set", "on_file_set", (VMachine*)_vm );
}
#endif // GTK_CHECK_VERSION( 2, 12, 0 )


//FALCON_FUNC FileChooserButton::new_with_backend( VMARG );


/*#
    @method new_with_dialog GtkFileChooserButton
    @brief Creates a GtkFileChooserButton widget which uses dialog as its file-picking window.
    @param dialog the widget to use as dialog
    @return a new button widget.

    Note that dialog must be a GtkDialog (or subclass) which implements the
    GtkFileChooser interface and must not have GTK_DIALOG_DESTROY_WITH_PARENT set.

    Also note that the dialog needs to have its confirmative button added with
    response GTK_RESPONSE_ACCEPT or GTK_RESPONSE_OK in order for the button to
    take over the file selected in the dialog.
 */
FALCON_FUNC FileChooserButton::new_with_dialog( VMARG )
{
    Item* i_dlg = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_dlg || i_dlg->isNil() || !i_dlg->isObject()
        || !IS_DERIVED( i_dlg, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* dlg = (GtkWidget*) COREGOBJECT( i_dlg )->getObject();
    GtkWidget* wdt = gtk_file_chooser_button_new_with_dialog( dlg );
    vm->retval( new Gtk::FileChooserButton(
            vm->findWKI( "GtkFileChooserButton" )->asClass(), (GtkFileChooserButton*) wdt ) );
}


/*#
    @method get_title GtkFileChooserButton
    @brief Retrieves the title of the browse dialog used by button.
    @return the browse dialog's title.
 */
FALCON_FUNC FileChooserButton::get_title( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( new String(
            gtk_file_chooser_button_get_title( (GtkFileChooserButton*)_obj ) ) );
}


/*#
    @method set_title GtkFileChooserButton
    @brief Modifies the title of the browse dialog used by button.
    @param title the new browse dialog title.
 */
FALCON_FUNC FileChooserButton::set_title( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    const char* title = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_button_set_title( (GtkFileChooserButton*)_obj, title );
}


/*#
    @method get_width_chars GtkFileChooserButton
    @brief Retrieves the width in characters of the button widget's entry and/or label.
    @return an integer width (in characters) that the button will use to size itself.
 */
FALCON_FUNC FileChooserButton::get_width_chars( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( gtk_file_chooser_button_get_width_chars( (GtkFileChooserButton*)_obj ) );
}


/*#
    @method set_width_chars GtkFileChooserButton
    @brief Sets the width (in characters) that button will use to n_chars.
    @param n_chars the new width, in characters.
 */
FALCON_FUNC FileChooserButton::set_width_chars( VMARG )
{
    Item* i_w = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_w || i_w->isNil() || !i_w->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_button_set_width_chars( (GtkFileChooserButton*)_obj, i_w->asInteger() );
}


/*#
    @method get_focus_on_click GtkFileChooserButton
    @brief Returns whether the button grabs focus when it is clicked with the mouse.
    @return true if the button grabs focus when it is clicked with the mouse
 */
FALCON_FUNC FileChooserButton::get_focus_on_click( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_button_get_focus_on_click(
            (GtkFileChooserButton*)_obj ) );
}


/*#
    @method set_focus_on_click GtkFileChooserButton
    @brief Sets whether the button will grab focus when it is clicked with the mouse.
    @param focus_on_click whether the button grabs focus when clicked with the mouse

    Making mouse clicks not grab focus is useful in places like toolbars where
    you don't want the keyboard focus removed from the main area of the application.
 */
FALCON_FUNC FileChooserButton::set_focus_on_click( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_button_set_focus_on_click(
            (GtkFileChooserButton*)_obj, i_bool->asBoolean() );
}


} // Gtk
} // Falcon
