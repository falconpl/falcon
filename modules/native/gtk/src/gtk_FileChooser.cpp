/**
 *  \file gtk_FileChooser.cpp
 */

#include "gtk_FileChooser.hpp"

#include "gtk_Widget.hpp"

/*#
   @beginmodule gtk
*/

namespace Falcon {
namespace Gtk {


/**
 *  \brief interface loader
 */
void FileChooser::clsInit( Falcon::Module* mod, Falcon::Symbol* cls )
{
    Gtk::MethodTab methods[] =
    {
    { "signal_confirm_overwrite",      &FileChooser::signal_confirm_overwrite },
    { "signal_current_folder_changed", &FileChooser::signal_current_folder_changed },
    { "signal_file_activated",         &FileChooser::signal_file_activated },
    { "signal_selection_changed",      &FileChooser::signal_selection_changed },
    { "signal_update_preview",         &FileChooser::signal_update_preview },
    { "set_action",             &FileChooser::set_action },
    { "get_action",             &FileChooser::get_action },
    { "set_local_only",         &FileChooser::set_local_only },
    { "get_local_only",         &FileChooser::get_local_only },
    { "set_select_multiple",    &FileChooser::set_select_multiple },
    { "get_select_multiple",    &FileChooser::get_select_multiple },
    { "set_show_hidden",        &FileChooser::set_show_hidden },
    { "get_show_hidden",        &FileChooser::get_show_hidden },
    { "set_do_overwrite_confirmation",&FileChooser::set_do_overwrite_confirmation },
    { "get_do_overwrite_confirmation",&FileChooser::get_do_overwrite_confirmation },
#if GTK_CHECK_VERSION( 2, 18, 0 )
    { "set_create_folders",     &FileChooser::set_create_folders },
    { "get_create_folders",     &FileChooser::get_create_folders },
#endif
    { "set_current_name",       &FileChooser::set_current_name },
    { "get_filename",           &FileChooser::get_filename },
    { "set_filename",           &FileChooser::set_filename },
    { "select_filename",        &FileChooser::select_filename },
    { "unselect_filename",      &FileChooser::unselect_filename },
    { "select_all",             &FileChooser::select_all },
    { "unselect_all",           &FileChooser::unselect_all },
    { "get_filenames",          &FileChooser::get_filenames },
    { "set_current_folder",     &FileChooser::set_current_folder },
    { "get_current_folder",     &FileChooser::get_current_folder },
    { "get_uri",                &FileChooser::get_uri },
    { "set_uri",                &FileChooser::set_uri },
    { "select_uri",             &FileChooser::select_uri },
    { "unselect_uri",           &FileChooser::unselect_uri },
    { "get_uris",               &FileChooser::get_uris },
    { "set_current_folder_uri", &FileChooser::set_current_folder_uri },
    { "get_current_folder_uri", &FileChooser::get_current_folder_uri },
    { "set_preview_widget",     &FileChooser::set_preview_widget },
    { "get_preview_widget",     &FileChooser::get_preview_widget },
    { "set_preview_widget_active",&FileChooser::set_preview_widget_active },
    { "get_preview_widget_active",&FileChooser::get_preview_widget_active },
    { "set_use_preview_label",  &FileChooser::set_use_preview_label },
    { "get_use_preview_label",  &FileChooser::get_use_preview_label },
    { "get_preview_filename",   &FileChooser::get_preview_filename },
    { "get_preview_uri",        &FileChooser::get_preview_uri },
    { "set_extra_widget",       &FileChooser::set_extra_widget },
    { "get_extra_widget",       &FileChooser::get_extra_widget },
#if 0
    { "add_filter",        &FileChooser:: },
    { "remove_filter",        &FileChooser:: },
    { "list_filters",        &FileChooser:: },
    { "set_filter",        &FileChooser:: },
    { "get_filter",        &FileChooser:: },
    { "add_shortcut_folder",        &FileChooser:: },
    { "remove_shortcut_folder",        &FileChooser:: },
    { "list_shortcut_folders",        &FileChooser:: },
    { "add_shortcut_folder_uri",        &FileChooser:: },
    { "remove_shortcut_folder_uri",        &FileChooser:: },
    { "list_shortcut_folder_uris",        &FileChooser:: },
#if GTK_CHECK_VERSION( 2, 14, 0 )
    { "get_current_folder_file",        &FileChooser:: },
    { "get_file",        &FileChooser:: },
    { "get_files",        &FileChooser:: },
    { "get_preview_file",        &FileChooser:: },
    { "select_file",        &FileChooser:: },
    { "set_current_folder_file",        &FileChooser:: },
    { "set_file",        &FileChooser:: },
    { "unselect_file",        &FileChooser:: },
#endif // GTK_CHECK_VERSION( 2, 14, 0 )
#endif
    { NULL, NULL }
    };

    for ( Gtk::MethodTab* meth = methods; meth->name; ++meth )
        mod->addClassMethod( cls, meth->name, meth->cb );
}


/*#
    @class GtkFileChooser
    @brief File chooser interface used by GtkFileChooserWidget and GtkFileChooserDialog

    GtkFileChooser is an interface that can be implemented by file selection widgets.
    In GTK+, the main objects that implement this interface are GtkFileChooserWidget,
    GtkFileChooserDialog, and GtkFileChooserButton. You do not need to write an object
    that implements the GtkFileChooser interface unless you are trying to adapt an
    existing file selector to expose a standard programming interface.

    [...]

 */

/*#
    @method signal_confirm_overwrite GtkFileChooser
    @brief This signal gets emitted whenever it is appropriate to present a confirmation dialog when the user has selected a file name that already exists.

    The signal only gets emitted when the file chooser is in GTK_FILE_CHOOSER_ACTION_SAVE
    mode.

    Most applications just need to turn on the "do-overwrite-confirmation" property
    (or call the gtk_file_chooser_set_do_overwrite_confirmation() function), and
    they will automatically get a stock confirmation dialog. Applications which
    need to customize this behavior should do that, and also connect to the
    "confirm-overwrite" signal.

    A signal handler for this signal must return a GtkFileChooserConfirmation value,
    which indicates the action to take. If the handler determines that the user wants
    to select a different filename, it should return
    GTK_FILE_CHOOSER_CONFIRMATION_SELECT_AGAIN. If it determines that the user is
    satisfied with his choice of file name, it should return
    GTK_FILE_CHOOSER_CONFIRMATION_ACCEPT_FILENAME. On the other hand, if it determines
    that the stock confirmation dialog should be used, it should return
    GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM.
 */
FALCON_FUNC FileChooser::signal_confirm_overwrite( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "confirm_overwrite",
        (void*) &FileChooser::on_confirm_overwrite, vm );
}


GtkFileChooserConfirmation
FileChooser::on_confirm_overwrite( GtkFileChooser* obj, gpointer _vm )
{
    GET_SIGNALS( obj );
    CoreSlot* cs = _signals->getChild( "confirm_overwrite", false );

    if ( !cs || cs->empty() )
        return GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM;

    VMachine* vm = (VMachine*) _vm;
    Iterator iter( cs );
    Item it;

    do
    {
        it = iter.getCurrent();

        if ( !it.isCallable() )
        {
            if ( !it.isComposed()
                || !it.asObject()->getMethod( "on_confirm_overwrite", it ) )
            {
                printf(
                "[GtkFileChooser::on_confirm_overwrite] invalid callback (expected callable)\n" );
                return GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM;
            }
        }
        vm->callItem( it, 0 );
        it = vm->regA();

        /*
         *  loop until the returned value is not 'confirm'
         */
        if ( !it.isNil() && it.isInteger() )
        {
            if ( it.asInteger() != GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM )
                return (GtkFileChooserConfirmation) it.asInteger();
            else
                iter.next();
        }
        else
        {
            printf(
            "[GtkFileChooser::on_confirm_overwrite] invalid callback (expected integer)\n" );
            return GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM;
        }
    }
    while ( iter.hasCurrent() );

    return GTK_FILE_CHOOSER_CONFIRMATION_CONFIRM;
}


/*#
    @method signal_current_folder_changed GtkFileChooser
    @brief This signal is emitted when the current folder in a GtkFileChooser changes.

    This can happen due to the user performing some action that changes folders,
    such as selecting a bookmark or visiting a folder on the file list. It can also
    happen as a result of calling a function to explicitly change the current folder
    in a file chooser.

    Normally you do not need to connect to this signal, unless you need to keep
    track of which folder a file chooser is showing.
 */
FALCON_FUNC FileChooser::signal_current_folder_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "current_folder_changed",
        (void*) &FileChooser::on_current_folder_changed, vm );
}


void FileChooser::on_current_folder_changed( GtkFileChooser* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "current_folder_changed",
        "on_current_folder_changed", (VMachine*)_vm );
}


/*#
    @method signal_file_activated GtkFileChooser
    @brief This signal is emitted when the user "activates" a file in the file chooser.

    This can happen by double-clicking on a file in the file list, or by pressing Enter.

    Normally you do not need to connect to this signal. It is used internally by
    GtkFileChooserDialog to know when to activate the default button in the dialog.
 */
FALCON_FUNC FileChooser::signal_file_activated( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "file_activated",
        (void*) &FileChooser::on_file_activated, vm );
}


void FileChooser::on_file_activated( GtkFileChooser* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "file_activated",
        "on_file_activated", (VMachine*)_vm );
}


/*#
    @method signal_selection_changed GtkFileChooser
    @brief This signal is emitted when there is a change in the set of selected files in a GtkFileChooser.

    This can happen when the user modifies the selection with the mouse or the keyboard,
    or when explicitly calling functions to change the selection.

    [...]
 */
FALCON_FUNC FileChooser::signal_selection_changed( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "selection_changed",
        (void*) &FileChooser::on_selection_changed, vm );
}


void FileChooser::on_selection_changed( GtkFileChooser* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "selection_changed",
        "on_selection_changed", (VMachine*)_vm );
}


/*#
    @method signal_update_preview GtkFileChooser
    @brief This signal is emitted when the preview in a file chooser should be regenerated.

    For example, this can happen when the currently selected file changes. You should
    use this signal if you want your file chooser to have a preview widget.

    Once you have installed a preview widget with gtk_file_chooser_set_preview_widget(),
    you should update it when this signal is emitted. You can use the functions
    gtk_file_chooser_get_preview_filename() or gtk_file_chooser_get_preview_uri() to
    get the name of the file to preview. Your widget may not be able to preview all
    kinds of files; your callback must call gtk_file_chooser_set_preview_widget_active()
    to inform the file chooser about whether the preview was generated successfully or not.
 */
FALCON_FUNC FileChooser::signal_update_preview( VMARG )
{
    NO_ARGS
    CoreGObject::get_signal( "update_preview",
        (void*) &FileChooser::on_update_preview, vm );
}


void FileChooser::on_update_preview( GtkFileChooser* obj, gpointer _vm )
{
    CoreGObject::trigger_slot( (GObject*) obj, "update_preview",
        "on_update_preview", (VMachine*)_vm );
}


/*#
    @method set_action GtkFileChooser
    @brief Sets the type of operation that the chooser is performing; the user interface is adapted to suit the selected action.
    @param action the action that the file selector is performing (GtkFileChooserAction).

    For example, an option to create a new folder might be shown if the action is
    GTK_FILE_CHOOSER_ACTION_SAVE but not if the action is GTK_FILE_CHOOSER_ACTION_OPEN.
 */
FALCON_FUNC FileChooser::set_action( VMARG )
{
    Item* i_act = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_act || i_act->isNil() || !i_act->isInteger() )
        throw_inv_params( "I" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_action(
            (GtkFileChooser*)_obj, (GtkFileChooserAction)i_act->asInteger() );
}


/*#
    @method get_action GtkFileChooser
    @brief Gets the type of operation that the file chooser is performing.
    @return the action that the file selector is performing (GtkFileChooserAction).
 */
FALCON_FUNC FileChooser::get_action( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (int64) gtk_file_chooser_get_action( (GtkFileChooser*)_obj ) );
}


/*#
    @method set_local_only GtkFileChooser
    @brief Sets whether only local files can be selected in the file selector.
    @param local_only true if only local files can be selected

    If local_only is TRUE (the default), then the selected file are files are guaranteed
    to be accessible through the operating systems native file file system and therefore
    the application only needs to worry about the filename functions in GtkFileChooser,
    like gtk_file_chooser_get_filename(), rather than the URI functions like
    gtk_file_chooser_get_uri().
 */
FALCON_FUNC FileChooser::set_local_only( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_local_only(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_local_only GtkFileChooser
    @brief Gets whether only local files can be selected in the file selector.
    @return true if only local files can be selected.
 */
FALCON_FUNC FileChooser::get_local_only( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_local_only( (GtkFileChooser*)_obj ) );
}


/*#
    @method set_select_multiple GtkFileChooser
    @brief Sets whether multiple files can be selected in the file selector.
    @param select_multiple true if multiple files can be selected.

    This is only relevant if the action is set to be GTK_FILE_CHOOSER_ACTION_OPEN
    or GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER.
 */
FALCON_FUNC FileChooser::set_select_multiple( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_select_multiple(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_select_multiple GtkFileChooser
    @brief Gets whether multiple files can be selected in the file selector.
    @return TRUE if multiple files can be selected.
 */
FALCON_FUNC FileChooser::get_select_multiple( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_select_multiple( (GtkFileChooser*)_obj ) );
}


/*#
    @method set_show_hidden GtkFileChooser
    @brief Sets whether hidden files and folders are displayed in the file selector.
    @param show_hidden true if hidden files and folders should be displayed.
 */
FALCON_FUNC FileChooser::set_show_hidden( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_show_hidden(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_show_hidden GtkFileChooser
    @brief Gets whether hidden files and folders are displayed in the file selector.
    @return true if hidden files and folders are displayed.
 */
FALCON_FUNC FileChooser::get_show_hidden( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_show_hidden( (GtkFileChooser*)_obj ) );
}


/*#
    @method set_do_overwrite_confirmation GtkFileChooser
    @brief Sets whether a file chooser in GTK_FILE_CHOOSER_ACTION_SAVE mode will present a confirmation dialog if the user types a file name that already exists.
    @param do_overwrite_confirmation whether to confirm overwriting in save mode

    This is FALSE by default.

    Regardless of this setting, the chooser will emit the "confirm_overwrite" signal
    when appropriate.

    If all you need is the stock confirmation dialog, set this property to TRUE.
    You can override the way confirmation is done by actually handling the "confirm_overwrite"
    signal; please refer to its documentation for the details.
 */
FALCON_FUNC FileChooser::set_do_overwrite_confirmation( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_do_overwrite_confirmation(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_do_overwrite_confirmation GtkFileChooser
    @brief Queries whether a file chooser is set to confirm for overwriting when the user types a file name that already exists.
    @return TRUE if the file chooser will present a confirmation dialog; FALSE otherwise.
 */
FALCON_FUNC FileChooser::get_do_overwrite_confirmation( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_do_overwrite_confirmation( (GtkFileChooser*)_obj ) );
}


#if GTK_CHECK_VERSION( 2, 18, 0 )
/*#
    @method set_create_folders GtkFileChooser
    @brief Sets whether file choser will offer to create new folders.
    @param create_folders TRUE if the New Folder button should be displayed

    This is only relevant if the action is not set to be GTK_FILE_CHOOSER_ACTION_OPEN.
 */
FALCON_FUNC FileChooser::set_create_folders( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_create_folders(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_create_folders GtkFileChooser
    @brief Gets whether file choser will offer to create new folders.
    @return TRUE if the New Folder button should be displayed.
 */
FALCON_FUNC FileChooser::get_create_folders( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_create_folders( (GtkFileChooser*)_obj ) );
}
#endif // GTK_CHECK_VERSION( 2, 18, 0 )


/*#
    @method set_current_name GtkFileChooser
    @brief Sets the current name in the file selector, as if entered by the user.
    @param the filename to use, as a UTF_8 string.

    Note that the name passed in here is a UTF_8 string rather than a filename.

    This function is meant for such uses as a suggested name in a "Save As..." dialog.
    If you want to preselect a particular existing file, you should use
    gtk_file_chooser_set_filename() or gtk_file_chooser_set_uri() instead.
    Please see the documentation for those functions for an example of using
    gtk_file_chooser_set_current_name() as well.
 */
FALCON_FUNC FileChooser::set_current_name( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_current_name( (GtkFileChooser*)_obj, nm );
}


/*#
    @method get_filename GtkFileChooser
    @brief Gets the filename for the currently selected file in the file selector.
    @return The currently selected filename, or nil if no file is selected, or the selected file can't be represented with a local filename.

    If multiple files are selected, one of the filenames will be returned at random.

    If the file chooser is in folder mode, this function returns the selected folder.
 */
FALCON_FUNC FileChooser::get_filename( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* nm = gtk_file_chooser_get_filename( (GtkFileChooser*)_obj );
    if ( nm )
    {
        vm->retval( new String( nm ) );
        g_free( nm );
    }
    else
        vm->retnil();
}


/*#
    @method set_filename GtkFileChooser
    @brief Sets filename as the current filename for the file chooser, by changing to the file's parent folder and actually selecting the file in list.
    @param filename the filename to set as current
    @return true if both the folder could be changed and the file was selected successfully, false otherwise.

    If the chooser is in GTK_FILE_CHOOSER_ACTION_SAVE mode, the file's base name will
    also appear in the dialog's file name entry.

    If the file name isn't in the current folder of chooser, then the current folder
    of chooser will be changed to the folder containing filename. This is equivalent
    to a sequence of gtk_file_chooser_unselect_all() followed by
    gtk_file_chooser_select_filename().

    Note that the file must exist, or nothing will be done except for the directory change.
 */
FALCON_FUNC FileChooser::set_filename( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_set_filename( (GtkFileChooser*)_obj, nm ) );
}


/*#
    @method select_filename GtkFileChooser
    @brief Selects a filename.
    @param filename the filename to select
    @return true if both the folder could be changed and the file was selected successfully, false otherwise

    If the file name isn't in the current folder of chooser, then the current folder
    of chooser will be changed to the folder containing filename.
 */
FALCON_FUNC FileChooser::select_filename( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_select_filename( (GtkFileChooser*)_obj, nm ) );
}


/*#
    @method unselect_filename GtkFileChooser
    @brief Unselects a currently selected filename.
    @param filename the filename to unselect

    If the filename is not in the current directory, does not exist, or is otherwise
    not currently selected, does nothing.
 */
FALCON_FUNC FileChooser::unselect_filename( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_unselect_filename( (GtkFileChooser*)_obj, nm );
}


/*#
    @method select_all GtkFileChooser
    @brief Selects all the files in the current folder of a file chooser.
 */
FALCON_FUNC FileChooser::select_all( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_select_all( (GtkFileChooser*)_obj );
}


/*#
    @method unselect_all GtkFileChooser
    @brief Unselects all the files in the current folder of a file chooser.
 */
FALCON_FUNC FileChooser::unselect_all( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_unselect_all( (GtkFileChooser*)_obj );
}


/*#
    @method get_filenames GtkFileChooser
    @brief Lists all the selected files and subfolders in the current folder of chooser.
    @return An array of strings

    The returned names are full absolute paths. If files in the current folder cannot
    be represented as local filenames they will be ignored.
 */
FALCON_FUNC FileChooser::get_filenames( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GSList* sl = gtk_file_chooser_get_filenames( (GtkFileChooser*)_obj );
    CoreArray* arr = new CoreArray( g_slist_length( sl ) );
    for( GSList* el = sl; el; el = el->next )
        arr->append( new String( (char*) el->data ) );
    vm->retval( arr );
}


/*#
    @method set_current_folder GtkFileChooser
    @brief Sets the current folder for chooser from a local filename.
    @param filename the full path of the new current folder
    @return true if the folder could be changed successfully, false otherwise.

    The user will be shown the full contents of the current folder, plus user
    interface elements for navigating to other folders.
 */
FALCON_FUNC FileChooser::set_current_folder( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* nm = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_set_current_folder( (GtkFileChooser*)_obj, nm ) );
}


/*#
    @method get_current_folder GtkFileChooser
    @brief Gets the current folder of chooser as a local filename.
    @return the full path of the current folder, or nil if the current path cannot be represented as a local filename.

    Note that this is the folder that the file chooser is currently displaying
    (e.g. "/home/username/Documents"), which is not the same as the currently_selected
    folder if the chooser is in GTK_FILE_CHOOSER_SELECT_FOLDER mode
    (e.g. "/home/username/Documents/selected_folder/".
    To get the currently_selected folder in that mode, use gtk_file_chooser_get_uri()
    as the usual way to get the selection.
 */
FALCON_FUNC FileChooser::get_current_folder( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* folder = gtk_file_chooser_get_current_folder( (GtkFileChooser*)_obj );
    if ( folder )
    {
        vm->retval( new String( folder ) );
        g_free( folder );
    }
    else
        vm->retnil();
}


/*#
    @method get_uri GtkFileChooser
    @brief Gets the URI for the currently selected file in the file selector.
    @return The currently selected URI, or NULL if no file is selected.

    If multiple files are selected, one of the filenames will be returned at random.

    If the file chooser is in folder mode, this function returns the selected folder.
 */
FALCON_FUNC FileChooser::get_uri( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* uri = gtk_file_chooser_get_uri( (GtkFileChooser*)_obj );
    if ( uri )
    {
        vm->retval( new String( uri ) );
        g_free( uri );
    }
    else
        vm->retnil();
}


/*#
    @method set_uri GtkFileChooser
    @brief Sets the file referred to by uri as the current file for the file chooser, by changing to the URI's parent folder and actually selecting the URI in the list.
    @param uri the URI to set as current
    @return true if both the folder could be changed and the URI was selected successfully, false otherwise.

    If the chooser is GTK_FILE_CHOOSER_ACTION_SAVE mode, the URI's base name will also
    appear in the dialog's file name entry.

    If the URI isn't in the current folder of chooser, then the current folder of
    chooser will be changed to the folder containing uri. This is equivalent to a
    sequence of gtk_file_chooser_unselect_all() followed by gtk_file_chooser_select_uri().
 */
FALCON_FUNC FileChooser::set_uri( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_set_uri( (GtkFileChooser*)_obj, uri ) );
}


/*#
    @method select_uri GtkFileChooser
    @brief Selects the file to by uri.
    @param uri the URI to set as current
    @return true if both the folder could be changed and the URI was selected successfully, false otherwise

    If the URI doesn't refer to a file in the current folder of chooser, then the
    current folder of chooser will be changed to the folder containing filename.
 */
FALCON_FUNC FileChooser::select_uri( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_select_uri( (GtkFileChooser*)_obj, uri ) );
}


/*#
    @method unselect_uri GtkFileChooser
    @brief Unselects the file referred to by uri.
    @param uri the URI to unselect

    If the file is not in the current directory, does not exist, or is otherwise
    not currently selected, does nothing.
 */
FALCON_FUNC FileChooser::unselect_uri( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_unselect_uri( (GtkFileChooser*)_obj, uri );
}


/*#
    @method get_uris GtkFileChooser
    @brief Lists all the selected files and subfolders in the current folder of chooser.
    @return An array of strings.

    The returned names are full absolute URIs.
 */
FALCON_FUNC FileChooser::get_uris( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GSList* sl = gtk_file_chooser_get_uris( (GtkFileChooser*)_obj );
    CoreArray* arr = new CoreArray( g_slist_length( sl ) );
    for( GSList* el = sl; el; el = el->next )
        arr->append( new String( (char*) el->data ) );
    vm->retval( arr );
}


/*#
    @method set_current_folder_uri GtkFileChooser
    @brief Sets the current folder for chooser from an URI.
    @param uri the URI for the new current folder
    @return true if the folder could be changed successfully, false otherwise

    The user will be shown the full contents of the current folder, plus user
    interface elements for navigating to other folders.
 */
FALCON_FUNC FileChooser::set_current_folder_uri( VMARG )
{
    Gtk::ArgCheck1 args( vm, "S" );
    char* uri = args.getCString( 0 );
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_set_current_folder_uri(
                (GtkFileChooser*)_obj, uri ) );
}


/*#
    @method get_current_folder_uri GtkFileChooser
    @brief Gets the current folder of chooser as an URI.
    @return the URI for the current folder (or nil).

    Note that this is the folder that the file chooser is currently displaying
    (e.g. "file:///home/username/Documents"), which is not the same as the
    currently_selected folder if the chooser is in GTK_FILE_CHOOSER_SELECT_FOLDER
    mode (e.g. "file:///home/username/Documents/selected_folder/".
    To get the currently_selected folder in that mode, use gtk_file_chooser_get_uri()
    as the usual way to get the selection.
 */
FALCON_FUNC FileChooser::get_current_folder_uri( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* uri = gtk_file_chooser_get_current_folder_uri( (GtkFileChooser*)_obj );
    if ( uri )
    {
        vm->retval( new String( uri ) );
        g_free( uri );
    }
    else
        vm->retnil();
}


/*#
    @method set_preview_widget GtkFileChooser
    @brief Sets an application_supplied widget to use to display a custom preview of the currently selected file.
    @param preview_widget widget for displaying preview.

    To implement a preview, after setting the preview widget, you connect to the
    "update_preview" signal, and call gtk_file_chooser_get_preview_filename() or
    gtk_file_chooser_get_preview_uri() on each change. If you can display a preview
    of the new file, update your widget and set the preview active using
    gtk_file_chooser_set_preview_widget_active(). Otherwise, set the preview inactive.

    When there is no application_supplied preview widget, or the application_supplied
    preview widget is not active, the file chooser may display an internally generated
    preview of the current file or it may display no preview at all
 */
FALCON_FUNC FileChooser::set_preview_widget( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget" );
    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_preview_widget( (GtkFileChooser*)_obj, wdt );
}


/*#
    @method get_preview_widget GtkFileChooser
    @brief Gets the current preview widget.
    @return the current preview widget, or nil.
 */
FALCON_FUNC FileChooser::get_preview_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_file_chooser_get_preview_widget( (GtkFileChooser*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


/*#
    @method set_preview_widget_active GtkFileChooser
    @brief Sets whether the preview widget set by gtk_file_chooser_set_preview_widget() should be shown for the current filename.
    @param active whether to display the user_specified preview widget

    When active is set to false, the file chooser may display an internally generated
    preview of the current file or it may display no preview at all.
    See gtk_file_chooser_set_preview_widget() for more details.
 */
FALCON_FUNC FileChooser::set_preview_widget_active( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_preview_widget_active(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_preview_widget_active GtkFileChooser
    @brief Gets whether the preview widget set by gtk_file_chooser_set_preview_widget() should be shown for the current filename.
    @return true if the preview widget is active for the current filename
 */
FALCON_FUNC FileChooser::get_preview_widget_active( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_preview_widget_active( (GtkFileChooser*)_obj ) );
}


/*#
    @method set_use_preview_label GtkFileChooser
    @brief Sets whether the file chooser should display a stock label with the name of the file that is being previewed.
    @param use_label whether to display a stock label with the name of the previewed file

    The default is true. Applications that want to draw the whole preview area
    themselves should set this to FALSE and display the name themselves in their preview widget.
 */
FALCON_FUNC FileChooser::set_use_preview_label( VMARG )
{
    Item* i_bool = vm->param( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !i_bool || i_bool->isNil() || !i_bool->isBoolean() )
        throw_inv_params( "B" );
#endif
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_use_preview_label(
            (GtkFileChooser*)_obj, i_bool->asBoolean() ? TRUE : FALSE );
}


/*#
    @method get_use_preview_label GtkFileChooser
    @brief Gets whether a stock label should be drawn with the name of the previewed file.
    @return TRUE if the file chooser is set to display a label with the name of the previewed file, FALSE otherwise
 */
FALCON_FUNC FileChooser::get_use_preview_label( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    vm->retval( (bool) gtk_file_chooser_get_use_preview_label( (GtkFileChooser*)_obj ) );
}


/*#
    @method get_preview_filename GtkFileChooser
    @brief Gets the filename that should be previewed in a custom preview widget.
    @return the filename to preview, or NULL if no file is selected, or if the selected file cannot be represented as a local filename.
 */
FALCON_FUNC FileChooser::get_preview_filename( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* nm = gtk_file_chooser_get_preview_filename( (GtkFileChooser*)_obj );
    if ( nm )
    {
        vm->retval( new String( nm ) );
        g_free( nm );
    }
    else
        vm->retnil();
}


/*#
    @method get_preview_uri GtkFileChooser
    @brief Gets the URI that should be previewed in a custom preview widget.
    @return the URI for the file to preview, or NULL if no file is selected.
 */
FALCON_FUNC FileChooser::get_preview_uri( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    char* uri = gtk_file_chooser_get_preview_uri( (GtkFileChooser*)_obj );
    if ( uri )
    {
        vm->retval( new String( uri ) );
        g_free( uri );
    }
    else
        vm->retnil();
}


/*#
    @method set_extra_widget GtkFileChooser
    @brief Sets an application_supplied widget to provide extra options to the user.
    @param extra_widget widget for extra options
 */
FALCON_FUNC FileChooser::set_extra_widget( VMARG )
{
    Gtk::ArgCheck0 args( vm, "GtkWidget" );
    CoreGObject* o_wdt = args.getCoreGObject( 0 );
#ifndef NO_PARAMETER_CHECK
    if ( !CoreObject_IS_DERIVED( o_wdt, GtkWidget ) )
        throw_inv_params( "GtkWidget" );
#endif
    GtkWidget* wdt = (GtkWidget*) o_wdt->getObject();
    MYSELF;
    GET_OBJ( self );
    gtk_file_chooser_set_extra_widget( (GtkFileChooser*)_obj, wdt );
}


/*#
    @method get_extra_widget GtkFileChooser
    @brief Gets the current preview widget.
    @return the current extra widget, or NULL
 */
FALCON_FUNC FileChooser::get_extra_widget( VMARG )
{
    NO_ARGS
    MYSELF;
    GET_OBJ( self );
    GtkWidget* wdt = gtk_file_chooser_get_extra_widget( (GtkFileChooser*)_obj );
    if ( wdt )
        vm->retval( new Gtk::Widget( vm->findWKI( "GtkWidget" )->asClass(), wdt ) );
    else
        vm->retnil();
}


#if 0
FALCON_FUNC FileChooser::add_filter( VMARG );
FALCON_FUNC FileChooser::remove_filter( VMARG );
FALCON_FUNC FileChooser::list_filters( VMARG );
FALCON_FUNC FileChooser::set_filter( VMARG );
FALCON_FUNC FileChooser::get_filter( VMARG );
FALCON_FUNC FileChooser::add_shortcut_folder( VMARG );
FALCON_FUNC FileChooser::remove_shortcut_folder( VMARG );
FALCON_FUNC FileChooser::list_shortcut_folders( VMARG );
FALCON_FUNC FileChooser::add_shortcut_folder_uri( VMARG );
FALCON_FUNC FileChooser::remove_shortcut_folder_uri( VMARG );
FALCON_FUNC FileChooser::list_shortcut_folder_uris( VMARG );
#if GTK_CHECK_VERSION( 2, 14, 0 )
FALCON_FUNC FileChooser::get_current_folder_file( VMARG );
FALCON_FUNC FileChooser::get_file( VMARG );
FALCON_FUNC FileChooser::get_files( VMARG );
FALCON_FUNC FileChooser::get_preview_file( VMARG );
FALCON_FUNC FileChooser::select_file( VMARG );
FALCON_FUNC FileChooser::set_current_folder_file( VMARG );
FALCON_FUNC FileChooser::set_file( VMARG );
FALCON_FUNC FileChooser::unselect_file( VMARG );
#endif // GTK_CHECK_VERSION( 2, 14, 0 )
#endif

} // Gtk
} // Falcon
